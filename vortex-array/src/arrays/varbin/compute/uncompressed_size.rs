// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::validity_uncompressed_size_in_bytes;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;

/// Measures the canonical representation from the offsets and validity.
#[derive(Debug)]
pub(crate) struct VarBinUncompressedSizeKernel;

impl DynAggregateKernel for VarBinUncompressedSizeKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<UncompressedSizeInBytes>() {
            return Ok(None);
        }

        let Some(array) = batch.as_opt::<VarBin>() else {
            return Ok(None);
        };

        Ok(Some(Scalar::from(uncompressed_size(array, ctx)?)))
    }
}

fn uncompressed_size(array: ArrayView<'_, VarBin>, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    let data_size = if validity.all_false() {
        0
    } else {
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        match_each_integer_ptype!(offsets.ptype(), |P| {
            referenced_data_size(offsets.as_slice::<P>(), &validity)
        })
    };
    let validity_size = validity_uncompressed_size_in_bytes(validity)?;

    u64::try_from(array.len())
        .ok()
        .and_then(|len| len.checked_mul(size_of::<BinaryView>() as u64))
        .and_then(|size| size.checked_add(u64::try_from(data_size).ok()?))
        .and_then(|size| size.checked_add(validity_size))
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}

/// Valid offsets are monotonic and bounded by the byte buffer length. The row lengths are
/// disjoint, so summing any subset of them fits in usize. Slices may start at a nonzero offset.
fn referenced_data_size<T: IntegerPType>(offsets: &[T], validity: &Mask) -> usize {
    let lengths = offsets.array_windows().map(|&[start, end]| {
        let (start, end): (usize, usize) = (start.as_(), end.as_());
        end - start
    });
    let out_of_line = |len| {
        if len > BinaryView::MAX_INLINED_SIZE {
            len
        } else {
            0
        }
    };
    match validity {
        Mask::AllTrue(_) => lengths.map(out_of_line).sum(),
        Mask::AllFalse(_) => 0,
        Mask::Values(values) => lengths
            .zip(values.bit_buffer().iter())
            .map(|(len, valid)| if valid { out_of_line(len) } else { 0 })
            .sum(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayVTable;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::canonical_uncompressed_size_in_bytes;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
    use crate::aggregate_fn::session::AggregateFnSessionExt;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBin;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::match_each_integer_ptype;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn matches_views_for_every_offset_type() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut rng = StdRng::seed_from_u64(823);
        for ptype in [
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
        ] {
            for _ in 0..32 {
                let mut end = 3;
                let offsets: Vec<_> = std::iter::once(end)
                    .chain(
                        [12, 13]
                            .into_iter()
                            .chain((0..6).map(|_| rng.next_u32() % 15))
                            .map(|len| {
                                end += len;
                                end
                            }),
                    )
                    .collect();
                let offsets = match_each_integer_ptype!(ptype, |P| {
                    PrimitiveArray::from_iter(offsets.iter().map(|&offset| {
                        <P as num_traits::NumCast>::from(offset)
                            .vortex_expect("the fixture offsets fit every integer width")
                    }))
                    .into_array()
                });
                let bytes = ByteBuffer::from(vec![b'a'; end as usize + 4]);
                for validity in [
                    Validity::NonNullable,
                    Validity::AllValid,
                    Validity::AllInvalid,
                    Validity::from_iter((0..8).map(|_| rng.next_u32() % 3 != 0)),
                ] {
                    let dtype = DType::Utf8(if matches!(validity, Validity::NonNullable) {
                        Nullability::NonNullable
                    } else {
                        Nullability::Nullable
                    });
                    let array = VarBinArray::try_new(
                        offsets.clone(),
                        bytes.clone(),
                        dtype,
                        validity,
                        &mut ctx,
                    )?
                    .into_array();
                    for input in [array.clone(), array.slice(1..7)?] {
                        assert!(input.is::<VarBin>());
                        let size = uncompressed_size_in_bytes(&input, &mut ctx)?;
                        let views = input.execute::<VarBinViewArray>(&mut ctx)?;
                        // Force the reference to measure the views even when execution carries
                        // the input's exact size statistic over to the canonical form.
                        assert_eq!(
                            size as u64,
                            canonical_uncompressed_size_in_bytes(
                                &Canonical::VarBinView(views),
                                &mut ctx,
                            )?,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    #[test]
    fn sizing_does_not_execute_the_varbin_array() -> VortexResult<()> {
        use crate::test_harness::trace::TraceOptions;
        use crate::test_harness::trace::TraceResolution;
        use crate::test_harness::trace::trace_op_with;

        let mut ctx = SESSION.create_execution_ctx();
        let array = VarBinArray::from_iter_nonnull(
            ["short", "a longer string"],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array();
        let traced = trace_op_with(
            TraceOptions {
                resolution: TraceResolution::Attempts,
            },
            || uncompressed_size_in_bytes(&array, &mut ctx),
        )?;
        assert_eq!(traced.output, 2 * 16 + "a longer string".len());
        let trace = traced.trace.to_string();
        assert!(
            !trace.contains("execute encoding=vortex.varbin("),
            "{trace}"
        );
        Ok(())
    }

    #[test]
    fn registers_varbin_size_kernel() {
        assert!(
            SESSION
                .aggregate_fns()
                .find_aggregate_kernel(VarBin.id(), UncompressedSizeInBytes.id())
                .is_some()
        );
    }
}
