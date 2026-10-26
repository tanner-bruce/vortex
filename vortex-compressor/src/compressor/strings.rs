// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Packed string trials and preparation of reusable dictionaries.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::stats::GenerateStatsOptions;
use crate::stats::StringStats;

/// Logical selection facts, computed without rebuilding values or codes.
pub(super) struct DictionaryAnalysis {
    /// Full-input statistics for the existing string estimator.
    pub stats: StringStats,
    /// Size of densely packed canonical logical strings.
    pub packed_nbytes: u64,
}

/// Reads canonical view bytes without an allocating scalar accessor.
fn value_bytes(values: &VarBinViewArray, slot: usize) -> &[u8] {
    let view = &values.views()[slot];
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let reference = view.as_view();
        &values.buffer(reference.buffer_index as usize)[reference.as_range()]
    }
}

/// Counts views, outlined occurrences and a mixed-validity bitmap.
fn packed_nbytes(len: usize, outlined: u64, null_count: usize) -> VortexResult<u64> {
    let views = u64::try_from(len)?.checked_mul(16);
    let validity = if null_count > 0 && null_count < len {
        u64::try_from(len.div_ceil(8))?
    } else {
        0
    };
    views
        .and_then(|bytes| bytes.checked_add(outlined))
        .and_then(|bytes| bytes.checked_add(validity))
        .ok_or_else(|| vortex_err!("Packed string byte size overflow"))
}

/// Counts effective nulls and the existing view prefixes before adaptive selection.
pub(super) fn analyze_dictionary(
    dict: &DictArray,
    opts: GenerateStatsOptions,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DictionaryAnalysis> {
    let _phase = tracing::debug_span!("dictionary_analyze").entered();
    let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
    let values = dict.values().clone().execute::<VarBinViewArray>(ctx)?;
    let code_mask = codes.validity()?.execute_mask(codes.len(), ctx)?;
    let value_mask = values.validity()?.execute_mask(values.len(), ctx)?;
    let mut seen = vec![
        false;
        if opts.count_distinct_values {
            values.len()
        } else {
            0
        }
    ];
    let mut prefixes = HashSet::with_capacity(if opts.count_distinct_values {
        values.len().min(dict.len() / 2)
    } else {
        0
    });
    let mut null_count = 0;
    let mut outlined = 0_u64;
    for (&code, code_valid) in codes.as_slice::<u32>().iter().zip(code_mask.iter()) {
        if !code_valid {
            null_count += 1;
            continue;
        }
        let slot = code as usize;
        if slot >= values.len() {
            return Err(vortex_err!(
                "String dictionary code {code} is out of bounds"
            ));
        }
        if !value_mask.value(slot) {
            null_count += 1;
            continue;
        }
        let view = &values.views()[slot];
        if !view.is_inlined() {
            outlined = outlined
                .checked_add(u64::from(view.as_view().size))
                .ok_or_else(|| vortex_err!("Packed string byte size overflow"))?;
        }
        if opts.count_distinct_values && !seen[slot] {
            seen[slot] = true;
            let prefix = view.as_u128().to_le_bytes();
            prefixes.insert(u64::from_le_bytes([
                prefix[0], prefix[1], prefix[2], prefix[3], prefix[4], prefix[5], prefix[6],
                prefix[7],
            ]));
        }
    }
    if opts.count_distinct_values && null_count > 0 {
        prefixes.insert(0);
    }
    Ok(DictionaryAnalysis {
        stats: StringStats::from_dictionary_counts(
            dict.len() - null_count,
            null_count,
            opts.count_distinct_values.then_some(prefixes.len()),
        )?,
        packed_nbytes: packed_nbytes(dict.len(), outlined, null_count)?,
    })
}

/// Rebuilds referenced unique values only after the dictionary scheme wins.
pub(super) fn prepare_dictionary(
    dict: &DictArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let _phase = tracing::debug_span!("dictionary_prepare").entered();
    let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
    let values = dict.values().clone().execute::<VarBinViewArray>(ctx)?;
    let code_mask = codes.validity()?.execute_mask(codes.len(), ctx)?;
    let value_mask = values.validity()?.execute_mask(values.len(), ctx)?;
    let mut translations = vec![None; values.len()];
    // Borrow source bytes while translating; the returned dictionary owns its packed bytes.
    let mut lookup: HashMap<&[u8], u32> = HashMap::with_capacity(values.len().min(dict.len()));
    let mut output_values = VarBinViewBuilder::with_capacity_in(
        DType::Utf8(Nullability::NonNullable),
        values.len().min(dict.len()),
        BufferAllocatorRef::statically_allocated(),
    );
    let mut output_codes = Vec::with_capacity(dict.len());
    let mut validity = BitBufferMut::with_capacity(dict.len());
    for (&code, code_valid) in codes.as_slice::<u32>().iter().zip(code_mask.iter()) {
        if !code_valid {
            output_codes.push(0);
            validity.append(false);
            continue;
        }
        let slot = code as usize;
        if slot >= values.len() {
            return Err(vortex_err!(
                "String dictionary code {code} is out of bounds"
            ));
        }
        if !value_mask.value(slot) {
            output_codes.push(0);
            validity.append(false);
            continue;
        }
        let translated = match translations[slot] {
            Some(translated) => translated,
            None => {
                let bytes = value_bytes(&values, slot);
                let next = u32::try_from(lookup.len())?;
                let translated = *lookup.entry(bytes).or_insert_with(|| {
                    output_values.append_value(bytes);
                    next
                });
                translations[slot] = Some(translated);
                translated
            }
        };
        output_codes.push(translated);
        validity.append(true);
    }
    let validity = Validity::from_bit_buffer(validity.freeze(), dict.dtype().nullability());
    let codes = PrimitiveArray::new(Buffer::from(output_codes), validity).into_array();
    let values = output_values.finish_into_varbinview().into_array();
    // SAFETY: a value is appended only when a valid input code first references its bytes.
    // That same iteration appends the translated valid output code. Deduplication only
    // reuses such an existing value; null input codes/values never append output values.
    Ok(unsafe { DictArray::try_new(codes, values)?.set_all_values_referenced(true) }.into_array())
}

/// Packs selected logical strings, dropping unrelated buffers and invalid view payloads.
pub(super) fn pack(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let _phase = tracing::debug_span!("dictionary_pack").entered();
    if let Some(dict) = array.as_opt::<Dict>() {
        let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
        let values = dict.values().clone().execute::<VarBinViewArray>(ctx)?;
        let code_mask = codes.validity()?.execute_mask(codes.len(), ctx)?;
        let value_mask = values.validity()?.execute_mask(values.len(), ctx)?;
        let mut builder = VarBinViewBuilder::with_capacity_in(
            array.dtype().clone(),
            array.len(),
            BufferAllocatorRef::statically_allocated(),
        );
        for (&code, valid) in codes.as_slice::<u32>().iter().zip(code_mask.iter()) {
            if !valid {
                builder.append_null();
                continue;
            }
            let slot = code as usize;
            if slot >= values.len() {
                return Err(vortex_err!(
                    "String dictionary code {code} is out of bounds"
                ));
            }
            if value_mask.value(slot) {
                builder.append_value(value_bytes(&values, slot));
            } else {
                builder.append_null();
            }
        }
        return Ok(builder.finish_into_varbinview().into_array());
    }
    let values = array.clone().execute::<VarBinViewArray>(ctx)?;
    let mask = values.validity()?.execute_mask(values.len(), ctx)?;
    let mut builder = VarBinViewBuilder::with_capacity_in(
        values.dtype().clone(),
        values.len(),
        BufferAllocatorRef::statically_allocated(),
    );
    for (slot, valid) in mask.iter().enumerate() {
        if valid {
            builder.append_value(value_bytes(&values, slot));
        } else {
            builder.append_null();
        }
    }
    Ok(builder.finish_into_varbinview().into_array())
}

/// Counts packed canonical bytes without allocating expanded string storage.
pub(super) fn logical_nbytes(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let values = array.clone().execute::<VarBinViewArray>(ctx)?;
    let mask = values.validity()?.execute_mask(values.len(), ctx)?;
    let mut outlined = 0_u64;
    for (&view, valid) in values.views().iter().zip(mask.iter()) {
        if valid && !view.is_inlined() {
            outlined = outlined
                .checked_add(u64::from(view.as_view().size))
                .ok_or_else(|| vortex_err!("Packed string byte size overflow"))?;
        }
    }
    packed_nbytes(values.len(), outlined, mask.false_count())
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::VarBinView;
    use vortex_array::arrays::dict::DictArrayExt;
    use vortex_array::assert_arrays_eq;

    use super::*;

    #[test]
    fn prepares_only_referenced_unique_values_and_effective_nulls() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values = VarBinViewArray::from_iter_nullable_str([
            Some("unused-long-string"),
            Some("same-prefix-long-a"),
            Some("same-prefix-long-a"),
            None,
            Some(""),
        ])
        .into_array();
        let codes =
            PrimitiveArray::from_option_iter([Some(2_u32), Some(1), Some(4), Some(3), None])
                .into_array();
        let dict = DictArray::try_new(codes, values)?;
        let opts = GenerateStatsOptions {
            count_distinct_values: true,
        };
        let facts = analyze_dictionary(&dict, opts, &mut ctx)?;
        let prepared = prepare_dictionary(&dict, &mut ctx)?;
        assert_arrays_eq!(&dict.clone().into_array(), &prepared, &mut ctx);
        let packed = pack(&dict.into_array(), &mut ctx)?;
        assert_eq!(facts.packed_nbytes, logical_nbytes(&packed, &mut ctx)?);
        let canonical = packed.as_::<VarBinView>().into_owned();
        let stats = StringStats::generate_opts(&canonical, opts, &mut ctx);
        assert_eq!(facts.stats.value_count(), stats.value_count());
        assert_eq!(facts.stats.null_count(), stats.null_count());
        assert_eq!(
            facts.stats.estimated_distinct_count(),
            stats.estimated_distinct_count()
        );
        let dict = prepared.as_::<Dict>();
        assert_eq!(dict.values().len(), 2);
        assert!(dict.has_all_values_referenced());
        dict.validate_all_values_referenced(&mut ctx)?;
        assert_eq!(
            dict.codes().dtype().as_ptype(),
            vortex_array::dtype::PType::U32
        );
        Ok(())
    }
    #[test]
    fn prepares_empty_null_and_sliced_dictionaries() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let opts = GenerateStatsOptions {
            count_distinct_values: true,
        };
        for codes in [
            vec![],
            vec![None, None],
            vec![Some(1_u32), Some(1)],
            vec![Some(0), None, Some(1)],
        ] {
            let values = VarBinViewArray::from_iter_nullable_str([
                Some(""),
                Some("repeated outlined string"),
                Some("unused"),
            ])
            .into_array();
            let dict =
                DictArray::try_new(PrimitiveArray::from_option_iter(codes).into_array(), values)?
                    .into_array();
            for input in [dict.clone(), dict.slice(0..dict.len().min(1))?] {
                let source = input.as_::<Dict>().into_owned();
                let facts = analyze_dictionary(&source, opts, &mut ctx)?;
                let prepared = prepare_dictionary(&source, &mut ctx)?;
                let packed = pack(&input, &mut ctx)?;
                assert_arrays_eq!(&packed, &prepared, &mut ctx);
                let prepared_dict = prepared.as_::<Dict>();
                assert!(prepared_dict.has_all_values_referenced());
                prepared_dict.validate_all_values_referenced(&mut ctx)?;
                assert_eq!(facts.packed_nbytes, logical_nbytes(&packed, &mut ctx)?);
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_referenced_out_of_bounds_codes() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let dict = DictArray::try_new(
            PrimitiveArray::from_iter([7_u32]).into_array(),
            VarBinViewArray::from_iter_str(["value"]).into_array(),
        )?;
        let error = analyze_dictionary(&dict, GenerateStatsOptions::default(), &mut ctx)
            .err()
            .expect("invalid code");
        assert!(error.to_string().contains("out of bounds"));
        Ok(())
    }
}
