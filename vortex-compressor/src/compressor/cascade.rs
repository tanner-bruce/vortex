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
use super::pins;
use super::pins::Lookup;
#[cfg(feature = "site-observer")]
use super::pins::ObservedSite;
use super::pins::PinTally;
use super::select::WinnerEstimate;
use super::strings;
use crate::scheme::CompressorContext;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::scheme::SitePath;
use crate::scheme::site;
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
        self.compress_at(array, SitePath::ROOT, exec_ctx)
    }

    /// Compresses an array like [`compress`](Self::compress) and returns how its pin lookups
    /// went. Without pins the tally is zero.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    pub fn compress_with_tally(
        &self,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, PinTally)> {
        let outer = pins::replace(PinTally::default());
        let compressed = self.compress(array, exec_ctx);
        let tally = pins::replace(outer);
        Ok((compressed?, tally))
    }

    /// Compresses an array like [`compress`](Self::compress) and returns the key and decision of
    /// every site the search decided, in visit order, for offline pin generation.
    ///
    /// # Errors
    ///
    /// Returns an error if canonicalization or compression fails.
    #[cfg(feature = "site-observer")]
    pub fn compress_observed(
        &self,
        array: &ArrayRef,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<(ArrayRef, Vec<ObservedSite>)> {
        let outer = pins::start_observing();
        let compressed = self.compress(array, exec_ctx);
        let observed = pins::stop_observing(outer);
        Ok((compressed?, observed))
    }

    /// Compresses an array at the structural site `site` with a fresh cascade.
    pub(super) fn compress_at(
        &self,
        array: &ArrayRef,
        site: SitePath,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let before_nbytes = array.nbytes();
        let span = trace::compress_span(array.len(), array.dtype(), before_nbytes);
        let _enter = span.enter();

        let compressed =
            self.compress_input(array, CompressorContext::new().with_site(site), exec_ctx)?;

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
                    &eligible,
                    compress_ctx,
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
        let site = compress_ctx.site();
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
                        self.compress_at(field, site.child_index(index), exec_ctx)
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
                let type_ids = self.compress_at(
                    union_array.type_ids(),
                    site.child(site::UNION_TYPE_IDS),
                    exec_ctx,
                )?;
                let children = union_array
                    .iter_children()
                    .enumerate()
                    .map(|(index, child)| {
                        self.compress_at(child, site.child_index(index), exec_ctx)
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
                let compressed_elems = self.compress_at(
                    fsl_array.elements(),
                    site.child(site::LIST_ELEMENTS),
                    exec_ctx,
                )?;

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
                let compressed_storage = self.compress_at(
                    ext_array.storage_array(),
                    site.child(site::EXTENSION_STORAGE),
                    exec_ctx,
                )?;
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
                let core_storage = self.compress_physical_slots(
                    variant_array.core_storage(),
                    site.child(site::VARIANT_CORE),
                    exec_ctx,
                )?;
                let shredded_site = site.child(site::VARIANT_SHREDDED);
                let shredded = variant_array
                    .shredded()
                    .map(|arr| {
                        // Avoid stack-overflow for variant shredded values
                        if arr.is::<Variant>() {
                            self.compress_physical_slots(arr, shredded_site, exec_ctx)
                        } else {
                            self.compress_at(arr, shredded_site, exec_ctx)
                        }
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
        // the pinned scheme's statistics.
        self.compress_planned(&eligible_schemes, compress_ctx, exec_ctx, |opts, _| {
            Ok((ArrayAndStats::new(array.clone(), opts), before_nbytes))
        })
    }

    /// Compresses one site's array: by its pinned decision if it has one, otherwise by the search
    /// over `eligible`. `build` makes the array with the given statistics options and returns its
    /// size before compression; a declined or failed replay calls it again for the search.
    ///
    /// A pinned scheme must be eligible at the site and admit the array by its own estimate,
    /// without sampling. `SequenceScheme`, for one, fails in `compress` on a chunk with nulls or
    /// one that is not an exact sequence. An error in the pinned scheme or any of its children
    /// falls back to the search too: the search still writes a valid array.
    fn compress_planned(
        &self,
        eligible: &[&'static dyn Scheme],
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
        mut build: impl FnMut(
            GenerateStatsOptions,
            &mut ExecutionCtx,
        ) -> VortexResult<(ArrayAndStats, u64)>,
    ) -> VortexResult<ArrayRef> {
        let planned = match self.pins {
            Some(pins) if !compress_ctx.is_sample() => match pins.get(&compress_ctx) {
                None => Planned::Unpinned,
                Some(None) => Planned::Canonical,
                Some(Some(scheme)) if !eligible.iter().any(|s| s.id() == scheme.id()) => {
                    Planned::Ineligible
                }
                Some(Some(scheme)) => {
                    let before = pins::current();
                    match self.replay(scheme, &compress_ctx, exec_ctx, &mut build) {
                        Ok(Some(compressed)) => return Ok(compressed),
                        // Counted as declined by `replay`.
                        Ok(None) => {}
                        Err(err) => {
                            trace::pin_replay_failed(scheme.id(), &compress_ctx, &err);
                            pins::fail(before, scheme.id());
                        }
                    }
                    Planned::Search
                }
            },
            _ => Planned::Search,
        };
        let eligible = match planned {
            Planned::Canonical => &[],
            Planned::Search | Planned::Unpinned | Planned::Ineligible => eligible,
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
            compress_ctx.with_merged_stats_options(opts),
            planned,
            exec_ctx,
        )
    }

    /// Replays the pinned `scheme` on the site's array with only its statistics, or returns
    /// `None`, counted as declined, if its estimate declines the array.
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
        // Like the search, ask no estimate about an empty, all-null or constant array: estimates
        // assume varying valid values (FoR's reads `max - min`, which an all-null chunk's stats
        // leave negative). These finish before the lookup and are not counted.
        let data = match self.finish_early(data, before_nbytes, &ctx, exec_ctx)? {
            Ok(finished) => return Ok(Some(finished)),
            Err(data) => data,
        };
        if !self.replay_admits(scheme, &data, ctx.clone(), exec_ctx)? {
            pins::count(Lookup::Declined);
            return Ok(None);
        }
        pins::count(Lookup::Pinned);
        self.compress_winner(
            data,
            scheme,
            WinnerEstimate::AlwaysUse,
            before_nbytes,
            ctx,
            exec_ctx,
        )
        .map(Some)
    }

    /// Selects a scheme over canonical data or a prepared logical string dictionary.
    fn choose_prepared(
        &self,
        data: ArrayAndStats,
        eligible_schemes: &[&'static dyn Scheme],
        before_nbytes: u64,
        compress_ctx: CompressorContext,
        planned: Planned,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let data = match self.finish_early(data, before_nbytes, &compress_ctx, exec_ctx)? {
            Ok(finished) => return Ok(finished),
            Err(data) => data,
        };
        match planned {
            Planned::Search => {}
            Planned::Unpinned => pins::count(Lookup::Searched),
            Planned::Ineligible => pins::count(Lookup::Ineligible),
            Planned::Canonical => {
                pins::count(Lookup::Pinned);
                return self.finish_uncompressed(data, &compress_ctx, exec_ctx);
            }
        }

        let chosen = if eligible_schemes.is_empty() {
            None
        } else {
            self.choose_best_scheme(eligible_schemes, &data, compress_ctx.clone(), exec_ctx)?
        };
        pins::observe(&compress_ctx, chosen.map(|(winner, _)| winner.id()));
        let Some((winner, winner_estimate)) = chosen else {
            return self.finish_uncompressed(data, &compress_ctx, exec_ctx);
        };
        self.compress_winner(
            data,
            winner,
            winner_estimate,
            before_nbytes,
            compress_ctx,
            exec_ctx,
        )
    }

    /// Finishes an empty, all-null or constant array without a scheme, or returns `data` back.
    ///
    /// Constant detection is skipped while compressing samples, because a constant sample does
    /// not imply that the full array is constant.
    fn finish_early(
        &self,
        data: ArrayAndStats,
        before_nbytes: u64,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Result<ArrayRef, ArrayAndStats>> {
        if data.array().is_empty() {
            return self
                .finish_uncompressed(data, compress_ctx, exec_ctx)
                .map(Ok);
        }
        let all_invalid = if data.array().is::<Dict>() {
            data.varbinview_stats(exec_ctx).null_count() as usize == data.array_len()
        } else {
            data.array().all_invalid(exec_ctx)?
        };
        if all_invalid {
            return Ok(Ok(ConstantArray::new(
                Scalar::null(data.array().dtype().clone()),
                data.array().len(),
            )
            .into_array()));
        }
        // Constant detection is built into the compressor: a constant leaf always short-circuits
        // scheme selection.
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
                Ok(Ok(compressed))
            } else {
                self.finish_uncompressed(data, compress_ctx, exec_ctx)
                    .map(Ok)
            };
        }
        Ok(Err(data))
    }

    /// Compresses `data` with the chosen `winner`, keeping the result only if it is smaller.
    fn compress_winner(
        &self,
        data: ArrayAndStats,
        winner: &'static dyn Scheme,
        winner_estimate: WinnerEstimate,
        before_nbytes: u64,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
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

/// How a site's array is compressed after its pin lookup, and how the lookup counts.
#[derive(Debug, Clone, Copy)]
enum Planned {
    /// Search, uncounted: no pins, a sample, or the search after a counted replay.
    Search,
    /// Search a site without a pin; counted as searched.
    Unpinned,
    /// Search a site whose pinned scheme is not eligible; counted as ineligible.
    Ineligible,
    /// Leave the array canonical, as its pin says; counted as pinned.
    Canonical,
}
