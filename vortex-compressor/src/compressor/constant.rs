// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Built-in constant detection and encoding.
//!
//! Constant arrays are not compressed through a pluggable [`Scheme`]: the compressor always
//! detects constant leaf arrays itself, before evaluating any registered scheme. Detection is
//! skipped while compressing samples, since a constant sample does not imply that the full array
//! is constant.
//!
//! [`Scheme`]: crate::scheme::Scheme

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::fns::is_constant::is_constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::MaskedArray;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::scheme::SchemeId;
use crate::stats::ArrayAndStats;

/// Synthetic scheme ID reported in traces when the compressor's built-in constant encoding wins.
pub(crate) const CONSTANT_SCHEME_ID: SchemeId = SchemeId {
    name: "vortex.compressor.constant",
};

/// Returns `true` if all valid values of the canonical array are equal, meaning the array can be
/// encoded by [`compress_constant`].
///
/// The caller must have already handled empty and all-null arrays.
///
/// Uses the cheapest available evidence per type: distinct counts when another scheme already
/// requested them, `O(1)` conclusions from type stats where possible, only the minimum and maximum
/// for integers whose stats are not cached yet, and otherwise a vectorized equality scan via
/// [`is_constant`].
///
/// Note that for types where the check falls through to [`is_constant`] (floats without distinct
/// counts, strings, binary, decimals, and extension types), arrays that contain any nulls are
/// reported as not constant, while stats-based checks detect constant valid values under nulls.
/// This mirrors the behavior of the per-type constant schemes this module replaced.
pub(crate) fn is_constant_for_compression(
    data: &ArrayAndStats,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if data.array().is::<Dict>() {
        // Source values can contain unused slots and duplicates. Match the canonical
        // string check, which rejects mixed nulls, and inspect only referenced values.
        let stats = data.varbinview_stats(exec_ctx);
        if stats.null_count() != 0 || stats.estimated_distinct_count().is_some_and(|n| n > 1) {
            return Ok(false);
        }
        return is_constant(data.array(), exec_ctx);
    }
    let dtype = data.array().dtype();

    if matches!(dtype, DType::Bool(_)) {
        return Ok(data.bool_stats(exec_ctx).is_constant());
    }

    if dtype.is_int() {
        if let Some(stats) = data.cached_integer_stats() {
            // Distinct counts are only computed when a registered scheme requested them.
            if let Some(distinct_count) = stats.distinct_count() {
                return Ok(distinct_count == 1);
            }

            // If max - min == 0 over the valid values, there is only one distinct value.
            return Ok(stats.erased().max_minus_min() == 0);
        }

        return Ok(integer_min_equals_max(data, exec_ctx));
    }

    if dtype.is_float() {
        let stats = data.float_stats(exec_ctx);

        if let Some(distinct_count) = stats.distinct_count() {
            return Ok(distinct_count == 1);
        }

        return is_constant(data.array(), exec_ctx);
    }

    if dtype.is_utf8() || dtype.is_binary() {
        let stats = data.varbinview_stats(exec_ctx);

        // The estimated distinct count is a lower bound on the actual distinct count, so a value
        // above 1 proves the array is not constant without scanning it.
        if stats.estimated_distinct_count().is_some_and(|c| c > 1) {
            return Ok(false);
        }

        return is_constant(data.array(), exec_ctx);
    }

    // Decimal, extension, and any other leaf type: fall back to the generic constant check.
    is_constant(data.array(), exec_ctx)
}

/// Returns `true` if the minimum and maximum of the valid values of an integer array are equal.
///
/// Computes only the array's `Min` and `Max` statistics, never the full [`IntegerStats`] with
/// their run count and distinct-value set, and leaves the stats cache empty: a scheme that later
/// asks for full stats still gets the distinct counts its options request. `Min` and `Max` stay
/// cached on the array exactly as full stats generation would have left them, so the serialized
/// array statistics do not change.
///
/// [`IntegerStats`]: crate::stats::IntegerStats
fn integer_min_equals_max(data: &ArrayAndStats, exec_ctx: &mut ExecutionCtx) -> bool {
    let stats = data.array().statistics();
    match_each_integer_ptype!(data.array_as_primitive().ptype(), |T| {
        let min = stats.compute_as::<T>(Stat::Min, exec_ctx);
        min.is_some() && min == stats.compute_as::<T>(Stat::Max, exec_ctx)
    })
}

/// Encodes an array whose valid values are all equal.
///
/// Returns a [`ConstantArray`], wrapped in a [`MaskedArray`] when the array has some nulls, or a
/// null [`ConstantArray`] when the array is all-null.
///
/// # Errors
///
/// Returns an error if computing validity or extracting the constant scalar fails.
pub(crate) fn compress_constant(
    source: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity = source.validity()?;
    let mask = validity.execute_mask(source.len(), ctx)?;

    let Some(first_valid) = mask.first() else {
        return Ok(
            ConstantArray::new(Scalar::null(source.dtype().clone()), source.len()).into_array(),
        );
    };

    let scalar = source.execute_scalar(first_valid, ctx)?;
    let const_arr = ConstantArray::new(scalar, source.len()).into_array();

    if mask.all_true() {
        Ok(const_arr)
    } else {
        Ok(MaskedArray::try_new(const_arr, validity)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::Masked;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::TemporalArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::NativePType;
    use vortex_array::expr::stats::Stat;
    use vortex_array::expr::stats::StatsProviderExt;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::is_constant_for_compression;
    use crate::CascadingCompressor;
    use crate::stats::ArrayAndStats;
    use crate::stats::GenerateStatsOptions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    /// Constant detection is built into the compressor, so it must work with no schemes at all.
    fn empty_compressor() -> CascadingCompressor {
        CascadingCompressor::new(Vec::new())
    }

    #[test]
    fn constant_int_compresses_without_schemes() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![7i64; 100], Validity::NonNullable).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn constant_int_with_nulls_compresses_to_masked_constant() -> VortexResult<()> {
        let validity =
            Validity::Array(BoolArray::from_iter((0..100).map(|i| i % 10 != 0)).into_array());
        let array = PrimitiveArray::new(buffer![7i64; 100], validity).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Masked>());
        Ok(())
    }

    #[test]
    fn constant_string_compresses_without_schemes() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_str(std::iter::repeat_n("hello", 100)).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn constant_bool_compresses_without_schemes() -> VortexResult<()> {
        let array = BoolArray::from_iter(std::iter::repeat_n(true, 100)).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn constant_decimal_compresses_without_schemes() -> VortexResult<()> {
        let array = DecimalArray::new(
            buffer![123_456i128; 100],
            DecimalDType::new(20, 2),
            Validity::NonNullable,
        )
        .into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    #[test]
    fn constant_timestamp_compresses_without_schemes() -> VortexResult<()> {
        let ts = PrimitiveArray::from_iter(std::iter::repeat_n(1_704_067_200_000i64, 100));
        let array = TemporalArray::new_timestamp(ts.into_array(), TimeUnit::Milliseconds, None)
            .into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(compressed.is::<Constant>());
        Ok(())
    }

    /// Distinct counting on, as when a dictionary or sparse scheme is eligible.
    const COUNT_DISTINCT: GenerateStatsOptions = GenerateStatsOptions {
        count_distinct_values: true,
    };

    /// Runs the constant check on `values` under `validity`, and checks that it left the stats
    /// cache empty.
    fn integer_constant<T: NativePType>(values: &[T], validity: Validity) -> VortexResult<bool> {
        let array = PrimitiveArray::new(Buffer::copy_from(values), validity).into_array();
        let data = ArrayAndStats::new(array, COUNT_DISTINCT);
        let mut ctx = SESSION.create_execution_ctx();

        let constant = is_constant_for_compression(&data, &mut ctx)?;
        assert!(data.cached_integer_stats().is_none());
        Ok(constant)
    }

    /// One test per integer width: all equal, one differing value, nulls, and all null.
    macro_rules! integer_width_tests {
        ($($name:ident: $T:ty),* $(,)?) => {$(
            #[test]
            fn $name() -> VortexResult<()> {
                let (low, high) = (<$T>::MIN, <$T>::MAX);
                assert!(integer_constant(&[low; 100], Validity::NonNullable)?);
                assert!(integer_constant(&[high; 100], Validity::NonNullable)?);

                // One differing value: first, in the 64-value head and in the tail after it.
                for at in [0, 50, 99] {
                    let mut values = [high; 100];
                    values[at] = low;
                    assert!(!integer_constant(&values, Validity::NonNullable)?);
                }

                // Differing values under nulls do not count; a valid one does.
                let mut values = [high; 100];
                values[10] = low;
                values[90] = low;
                let validity = Validity::from_iter((0..100).map(|i| i != 10 && i != 90));
                assert!(integer_constant(&values, validity)?);
                let validity = Validity::from_iter((0..100).map(|i| i != 10));
                assert!(!integer_constant(&values, validity)?);

                // All null finishes before the check, as a null constant.
                let all_null =
                    PrimitiveArray::new(Buffer::copy_from([high; 100]), Validity::AllInvalid)
                        .into_array();
                let mut ctx = SESSION.create_execution_ctx();
                let compressed = empty_compressor().compress(&all_null, &mut ctx)?;
                assert!(compressed.is::<Constant>());
                assert!(!compressed.is_valid(0, &mut ctx)?);
                Ok(())
            }
        )*};
    }

    integer_width_tests! {
        constant_check_u8: u8,
        constant_check_u16: u16,
        constant_check_u32: u32,
        constant_check_u64: u64,
        constant_check_i8: i8,
        constant_check_i16: i16,
        constant_check_i32: i32,
        constant_check_i64: i64,
    }

    /// The check caches `Min` and `Max` on the array, as full stats generation did, so the
    /// serialized statistics do not change, and leaves the stats cache to a later full request,
    /// which still counts distinct values.
    #[test]
    fn integer_check_caches_min_max_and_leaves_full_stats_to_schemes() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([3i32, 1, 3, 2, 3]).into_array();
        let data = ArrayAndStats::new(array.clone(), COUNT_DISTINCT);
        let mut ctx = SESSION.create_execution_ctx();

        assert!(!is_constant_for_compression(&data, &mut ctx)?);
        assert!(data.cached_integer_stats().is_none());
        let statistics = array.statistics();
        assert_eq!(statistics.get_as::<i32>(Stat::Min).as_exact(), Some(1));
        assert_eq!(statistics.get_as::<i32>(Stat::Max).as_exact(), Some(3));

        assert_eq!(data.integer_stats(&mut ctx).distinct_count(), Some(3));
        Ok(())
    }

    /// Full stats that a scheme already generated answer the check.
    #[test]
    fn integer_check_reads_cached_full_stats() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        for (values, constant) in [([5u16; 4], true), ([5, 5, 6, 5], false)] {
            let array = PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable);
            let data = ArrayAndStats::new(array.into_array(), COUNT_DISTINCT);
            let distinct = data.integer_stats(&mut ctx).distinct_count();
            assert_eq!(distinct, Some(if constant { 1 } else { 2 }));

            assert_eq!(is_constant_for_compression(&data, &mut ctx)?, constant);
        }
        Ok(())
    }

    #[test]
    fn non_constant_int_is_left_canonical_without_schemes() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter(0..100i64).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let compressed = empty_compressor().compress(&array, &mut ctx)?;
        assert!(!compressed.is::<Constant>());
        assert_eq!(compressed.dtype(), array.dtype());
        Ok(())
    }
}
