// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core cascading compression flow.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::CanonicalValidity;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::Masked;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::Variant;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::arrays::masked::MaskedArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use super::CascadingCompressor;
use super::constant;
use super::plan;
use super::plan::Planned;
use super::plan::site;
use super::plan::with_site;
use super::select::WinnerEstimate;
use super::strings;
use crate::scheme::CompressorContext;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;
use crate::stats::GenerateStatsOptions;
use crate::trace;

impl CascadingCompressor {
    /// Compresses an array using cascading adaptive compression.
    ///
    /// First canonicalizes and compacts the array, then applies optimal compression schemes.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress(
        &self,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let before_nbytes = array.nbytes();
        let span = trace::compress_span(array.len(), array.dtype(), before_nbytes);
        let _enter = span.enter();

        let compressed = match &self.plan {
            Some(plan) if plan::is_top_level() => {
                let before = plan::thread_fallbacks();
                let compressed = self.compress_input(array, CompressorContext::new(), exec_ctx)?;
                plan.chunk(plan::thread_fallbacks() != before);
                compressed
            }
            _ => self.compress_input(array, CompressorContext::new(), exec_ctx)?,
        };

        trace::record_compress_outcome(&span, before_nbytes, compressed.nbytes());

        Ok(compressed)
    }

    /// Compresses a child array produced by a cascading scheme.
    ///
    /// If the cascade budget is exhausted, the canonical array is returned as-is. Otherwise, the
    /// child context is created by descending and recording the parent scheme + child index, and
    /// compression proceeds normally.
    ///
    /// # Errors
    ///
    /// Returns an error if compression fails.
    pub fn compress_child(
        &self,
        child: &ArrayRef,
        parent_ctx: &CompressorContext,
        parent_id: SchemeId,
        child_index: usize,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if parent_ctx.finished_cascading() {
            trace::cascade_exhausted(parent_id, child_index);
            return Ok(child.clone());
        }

        let child_ctx = parent_ctx
            .clone()
            .descend_with_scheme(parent_id, child_index);
        self.compress_input(child, child_ctx, exec_ctx)
    }

    /// Preserves supported Utf8 dictionaries before the canonical dispatch.
    fn compress_input(
        &self,
        array: &ArrayRef,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if self.string_dictionary_reuse
            && array.dtype().is_utf8()
            && let Some(dict) = array.as_opt::<Dict>()
            && dict
                .codes()
                .as_opt::<Primitive>()
                .is_some_and(|codes| codes.ptype() == PType::U32)
        {
            let mut eligible = Vec::new();
            let mut supported = true;
            for scheme in &self.schemes {
                if self.is_excluded(*scheme, &compress_ctx) {
                    continue;
                }
                match scheme.matches_string_dictionary() {
                    Some(true) => eligible.push(*scheme),
                    Some(false) => {}
                    None => {
                        supported = false;
                        break;
                    }
                }
            }
            if supported {
                let dict = dict.into_owned();
                let mut compress_ctx = compress_ctx;
                compress_ctx.logical_strings = true;
                return self.compress_planned(
                    eligible,
                    &compress_ctx,
                    exec_ctx,
                    |opts, exec_ctx| {
                        let prepared = strings::analyze_dictionary(&dict, opts, exec_ctx)?;
                        let data =
                            ArrayAndStats::new_dictionary(prepared.array, opts, prepared.stats);
                        Ok((data, prepared.packed_nbytes))
                    },
                );
            }
        }
        let canonical = array.clone().execute::<CanonicalValidity>(exec_ctx)?.0;
        self.compress_canonical(canonical.compact(exec_ctx)?, compress_ctx, exec_ctx)
    }

    /// Compresses a canonical array by dispatching to type-specific logic.
    ///
    /// # Errors
    ///
    /// Returns an error if compression of any sub-array fails.
    pub(super) fn compress_canonical(
        &self,
        array: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        match array {
            Canonical::Null(null_array) => Ok(null_array.into_array()),
            Canonical::Bool(bool_array) => {
                self.choose_and_compress(Canonical::Bool(bool_array), compress_ctx, exec_ctx)
            }
            Canonical::Primitive(primitive) => {
                self.choose_and_compress(Canonical::Primitive(primitive), compress_ctx, exec_ctx)
            }
            Canonical::Decimal(decimal) => {
                self.choose_and_compress(Canonical::Decimal(decimal), compress_ctx, exec_ctx)
            }
            Canonical::Struct(struct_array) => {
                let fields = struct_array
                    .iter_unmasked_fields()
                    .enumerate()
                    .map(|(index, field)| {
                        with_site(index as u32, || self.compress(field, exec_ctx))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(StructArray::try_new(
                    struct_array.names().clone(),
                    fields,
                    struct_array.len(),
                    struct_array.validity()?,
                )?
                .into_array())
            }
            Canonical::Union(union_array) => {
                let type_ids = with_site(site::UNION_TYPE_IDS, || {
                    self.compress(union_array.type_ids(), exec_ctx)
                })?;
                let children = union_array
                    .iter_children()
                    .enumerate()
                    .map(|(index, child)| {
                        with_site(index as u32, || self.compress(child, exec_ctx))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(
                    UnionArray::try_new(type_ids, union_array.variants().clone(), children)?
                        .into_array(),
                )
            }
            Canonical::List(list_view_array) => {
                if list_view_array.is_zero_copy_to_list() || list_view_array.elements().is_empty() {
                    let list_array = list_from_list_view(list_view_array, exec_ctx)?;
                    self.compress_list_array(list_array, compress_ctx, exec_ctx)
                } else {
                    self.compress_list_view_array(list_view_array, compress_ctx, exec_ctx)
                }
            }
            Canonical::Map(map_array) => self.compress_map_array(map_array, compress_ctx, exec_ctx),
            Canonical::FixedSizeList(fsl_array) => {
                let compressed_elems = with_site(site::LIST_ELEMENTS, || {
                    self.compress(fsl_array.elements(), exec_ctx)
                })?;

                Ok(FixedSizeListArray::try_new(
                    compressed_elems,
                    fsl_array.list_size(),
                    fsl_array.validity()?,
                    fsl_array.len(),
                )?
                .into_array())
            }
            Canonical::VarBinView(varbinview) => {
                self.choose_and_compress(Canonical::VarBinView(varbinview), compress_ctx, exec_ctx)
            }
            Canonical::Extension(ext_array) => {
                // Try scheme-based compression first.
                let scheme_compressed = self.choose_and_compress(
                    Canonical::Extension(ext_array.clone()),
                    compress_ctx,
                    exec_ctx,
                )?;

                // A constant extension array (that might be masked) is already in its terminal
                // representation, and compressing the storage separately cannot do better.
                if scheme_compressed.is::<Constant>() {
                    return Ok(scheme_compressed);
                }
                if let Some(masked) = scheme_compressed.as_opt::<Masked>()
                    && masked.child().is::<Constant>()
                {
                    return Ok(scheme_compressed);
                }

                // Also compress the underlying storage array. Some extension schemes can beat the
                // extension storage but still lose to ordinary storage compression.
                let compressed_storage = with_site(site::EXTENSION_STORAGE, || {
                    self.compress(ext_array.storage_array(), exec_ctx)
                })?;
                let storage_compressed =
                    ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_storage)
                        .into_array();

                if scheme_compressed.nbytes() < storage_compressed.nbytes() {
                    Ok(scheme_compressed)
                } else {
                    Ok(storage_compressed)
                }
            }
            Canonical::Variant(variant_array) => {
                let core_storage = with_site(site::VARIANT_SLOTS, || {
                    self.compress_physical_slots(variant_array.core_storage(), exec_ctx)
                })?;
                let shredded = variant_array
                    .shredded()
                    .map(|arr| {
                        with_site(site::VARIANT_SLOTS + 1, || {
                            // Avoid stack-overflow for variant shredded values
                            if arr.is::<Variant>() {
                                self.compress_physical_slots(arr, exec_ctx)
                            } else {
                                self.compress(arr, exec_ctx)
                            }
                        })
                    })
                    .transpose()?;

                Ok(VariantArray::try_new(core_storage, shredded)?.into_array())
            }
        }
    }

    /// The main scheme-selection entry point for a single leaf array.
    ///
    /// Filters allowed schemes by [`matches`] and exclusion rules, merges their [`stats_options`]
    /// into a single [`GenerateStatsOptions`], and picks the winner by estimated compression
    /// ratio.
    ///
    /// If a winner is found and its compressed output is actually smaller, that output is
    /// returned. Otherwise, the original array is returned unchanged.
    ///
    /// Empty, all-null, and constant arrays are handled by the compressor itself before any
    /// scheme evaluation (constant detection is skipped while compressing samples).
    ///
    /// [`matches`]: Scheme::matches
    /// [`stats_options`]: Scheme::stats_options
    fn choose_and_compress(
        &self,
        canonical: Canonical,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let eligible_schemes: Vec<&'static dyn Scheme> = self
            .schemes
            .iter()
            .copied()
            .filter(|s| s.matches(&canonical) && !self.is_excluded(*s, &compress_ctx))
            .collect();

        let array: ArrayRef = canonical.into();
        let logical_strings = self.string_dictionary_reuse
            && array.dtype().is_utf8()
            && eligible_schemes
                .iter()
                .all(|s| s.matches_string_dictionary() == Some(true));
        let before_nbytes = if logical_strings {
            strings::logical_nbytes(&array, exec_ctx)?
        } else {
            array.nbytes()
        };
        let mut compress_ctx = compress_ctx;
        compress_ctx.logical_strings = logical_strings;
        // A replayed site keeps the full list's logical-strings choice above, but generates only
        // the replayed winner's statistics.
        self.compress_planned(eligible_schemes, &compress_ctx, exec_ctx, |opts, _| {
            Ok((ArrayAndStats::new(array.clone(), opts), before_nbytes))
        })
    }

    /// Compresses one site's array under its plan decision. `build` makes the array with the
    /// given statistics and its size before compression.
    ///
    /// A replayed winner must still admit the array by its own estimate, without sampling:
    /// `SequenceScheme`, for one, fails in `compress` on a chunk with nulls or one that is not an
    /// exact sequence. A declined replay counts as a plan miss. A replay that fails anyway, by an
    /// error or a panic in the winner or any of its children, counts as a fallback. Either way the
    /// array then gets a full selection over every eligible scheme and their statistics.
    fn compress_planned(
        &self,
        eligible: Vec<&'static dyn Scheme>,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
        mut build: impl FnMut(
            GenerateStatsOptions,
            &mut ExecutionCtx,
        ) -> VortexResult<(ArrayAndStats, u64)>,
    ) -> VortexResult<ArrayRef> {
        let mut planned = self.plan_lookup(compress_ctx, &eligible);
        if let (Planned::Replay(Some(scheme)), Some(plan)) = (&planned, &self.plan) {
            let scheme = *scheme;
            let replay =
                plan::catch_replay(|| self.replay(scheme, compress_ctx, exec_ctx, &mut build));
            match replay {
                Ok(Ok(Some(compressed))) => return Ok(compressed),
                Ok(Ok(None)) => plan.miss(),
                Ok(Err(err)) => plan.fallback(scheme.id(), &format!("error: {err}")),
                Err(panic) => plan.fallback(scheme.id(), &panic),
            }
            planned = Planned::Off;
        }
        // A site replayed as canonical has no candidates.
        let eligible = if matches!(planned, Planned::Replay(None)) {
            Vec::new()
        } else {
            eligible
        };
        let opts = eligible
            .iter()
            .fold(GenerateStatsOptions::default(), |acc, s| {
                acc.merge(s.stats_options())
            });
        let (data, before_nbytes) = build(opts, exec_ctx)?;
        self.choose_prepared(
            data,
            eligible,
            before_nbytes,
            compress_ctx.clone().with_merged_stats_options(opts),
            planned,
            exec_ctx,
        )
    }

    /// Replays `scheme` on the site's array with only its statistics, or `None` if its estimate
    /// declines the array.
    fn replay(
        &self,
        scheme: &'static dyn Scheme,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
        build: &mut impl FnMut(
            GenerateStatsOptions,
            &mut ExecutionCtx,
        ) -> VortexResult<(ArrayAndStats, u64)>,
    ) -> VortexResult<Option<ArrayRef>> {
        let opts = GenerateStatsOptions::default().merge(scheme.stats_options());
        let (data, before_nbytes) = build(opts, exec_ctx)?;
        let ctx = compress_ctx.clone().with_merged_stats_options(opts);
        // Like a full selection, ask no estimate about an empty, all-null or constant array:
        // `choose_prepared` finishes those before any scheme runs, and estimates assume varying
        // valid values (FoR's reads `max - min`, which an all-null chunk's stats leave negative).
        let finished_early = Self::no_valid_values(&data, exec_ctx)?
            || constant::is_constant_for_compression(&data, exec_ctx)?;
        if !finished_early && !self.replay_admits(scheme, &data, ctx.clone(), exec_ctx)? {
            return Ok(None);
        }
        self.choose_prepared(
            data,
            vec![scheme],
            before_nbytes,
            ctx,
            Planned::Replay(Some(scheme)),
            exec_ctx,
        )
        .map(Some)
    }

    /// Whether `data` is empty or all null: arrays `choose_prepared` finishes without a scheme.
    fn no_valid_values(data: &ArrayAndStats, exec_ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        if data.array().is_empty() {
            return Ok(true);
        }
        Ok(if data.array().is::<Dict>() {
            data.varbinview_stats(exec_ctx).null_count() as usize == data.array_len()
        } else {
            data.array().all_invalid(exec_ctx)?
        })
    }

    /// The plan's decision for this site, or [`Planned::Off`] without a plan.
    fn plan_lookup(
        &self,
        compress_ctx: &CompressorContext,
        eligible: &[&'static dyn Scheme],
    ) -> Planned {
        match &self.plan {
            Some(plan) => plan.lookup(compress_ctx, eligible),
            None => Planned::Off,
        }
    }

    /// Selects a scheme over canonical data or a prepared logical string dictionary.
    fn choose_prepared(
        &self,
        data: ArrayAndStats,
        eligible_schemes: Vec<&'static dyn Scheme>,
        before_nbytes: u64,
        compress_ctx: CompressorContext,
        planned: Planned,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if data.array().is_empty() {
            return self.finish_uncompressed(data, &compress_ctx, exec_ctx);
        }
        if Self::no_valid_values(&data, exec_ctx)? {
            return Ok(ConstantArray::new(
                Scalar::null(data.array().dtype().clone()),
                data.array().len(),
            )
            .into_array());
        }
        // Constant detection is built into the compressor: a constant leaf always short-circuits
        // scheme selection. Samples are exempt because a constant sample does not imply that the
        // full array is constant.
        if !compress_ctx.is_sample() && constant::is_constant_for_compression(&data, exec_ctx)? {
            let _winner_span =
                trace::winner_compress_span(constant::CONSTANT_SCHEME_ID, before_nbytes).entered();
            let compressed = constant::compress_constant(data.array(), exec_ctx)?;

            let after_nbytes = compressed.nbytes();
            let actual_ratio =
                (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);
            let accepted = after_nbytes < before_nbytes;
            trace::record_winner_compress_result(after_nbytes, None, actual_ratio, accepted);

            return if accepted {
                Ok(compressed)
            } else {
                self.finish_uncompressed(data, &compress_ctx, exec_ctx)
            };
        }

        if eligible_schemes.is_empty() {
            return self.finish_uncompressed(data, &compress_ctx, exec_ctx);
        }

        let chosen = match planned {
            Planned::Replay(Some(scheme)) => Some((scheme, WinnerEstimate::AlwaysUse)),
            Planned::Replay(None) => None,
            Planned::Off => {
                self.choose_best_scheme(&eligible_schemes, &data, compress_ctx.clone(), exec_ctx)?
            }
            Planned::Record(key) => {
                let chosen = self.choose_best_scheme(
                    &eligible_schemes,
                    &data,
                    compress_ctx.clone(),
                    exec_ctx,
                )?;
                if let Some(plan) = &self.plan {
                    plan.record(key, chosen.map(|(scheme, _)| scheme.id()));
                }
                chosen
            }
        };
        let Some((winner, winner_estimate)) = chosen else {
            return self.finish_uncompressed(data, &compress_ctx, exec_ctx);
        };

        let packed_full =
            compress_ctx.logical_strings && winner.id() != crate::builtins::StringDictScheme.id();
        let data = if packed_full {
            ArrayAndStats::new(
                strings::pack(data.array(), exec_ctx)?,
                compress_ctx.merged_stats_options(),
            )
        } else if compress_ctx.logical_strings
            && let Some(dict) = data.array().as_opt::<Dict>()
        {
            let stats = (*data.varbinview_stats(exec_ctx)).clone();
            ArrayAndStats::new_dictionary(
                strings::prepare_dictionary(&dict.into_owned(), exec_ctx)?,
                compress_ctx.merged_stats_options(),
                stats,
            )
        } else {
            data
        };

        // Run the winning scheme's `compress`. On failure, emit an ERROR event carrying the
        // scheme name and cascade history before propagating.
        let error_ctx = trace::enabled_error_context(&compress_ctx);
        let _winner_span = trace::winner_compress_span(winner.id(), before_nbytes).entered();
        let compressed = winner
            .compress(self, &data, compress_ctx.clone(), exec_ctx)
            .inspect_err(|err| {
                // NB: this is the only way we can tell which scheme panicked / bailed on their
                // data, especially for third-party schemes where the error site may not carry any
                // compressor context.
                trace::scheme_compress_failed(winner.id(), before_nbytes, error_ctx.as_ref(), err);
            })?;

        let after_nbytes = compressed.nbytes();
        let actual_ratio = (after_nbytes != 0).then(|| before_nbytes as f64 / after_nbytes as f64);

        let accepted = after_nbytes < before_nbytes;

        trace::record_winner_compress_result(
            after_nbytes,
            winner_estimate.trace_ratio(),
            actual_ratio,
            accepted,
        );

        if accepted {
            Ok(compressed)
        } else if packed_full {
            Ok(data.into_array())
        } else {
            self.finish_uncompressed(data, &compress_ctx, exec_ctx)
        }
    }
    /// Returns the canonical fallback, packing logical strings only when needed.
    fn finish_uncompressed(
        &self,
        data: ArrayAndStats,
        ctx: &CompressorContext,
        exec: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if ctx.logical_strings {
            strings::pack(data.array(), exec)
        } else {
            Ok(data.into_array())
        }
    }
}
