// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;

use super::MinMaxPartial;
use super::MinMaxResult;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::fns::bounded_max::BoundedMax;
use crate::aggregate_fn::fns::bounded_min::BoundedMin;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::kernels::DynAggregateKernel;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;

pub(super) fn accumulate_varbinview(
    partial: &mut MinMaxPartial,
    array: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    partial.merge(varbin_compute_min_max(array, array.dtype(), ctx)?);
    Ok(())
}

fn varbin_compute_min_max(
    array: &VarBinViewArray,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<MinMaxResult>> {
    let mask = array
        .validity()?
        .execute_mask(array.len(), ctx)?
        .to_bit_buffer();
    let views = array.views();
    let buffers = array
        .data_buffers()
        .iter()
        .map(|b| b.as_host())
        .collect::<Vec<_>>();
    let minmax = views
        .iter()
        .zip(mask.iter())
        .filter(|(_, v)| *v)
        .map(|(view, _)| {
            if view.is_inlined() {
                view.as_inlined().value()
            } else {
                let view_ref = view.as_view();
                &buffers[view_ref.buffer_index as usize][view_ref.as_range()]
            }
        })
        .minmax();
    Ok(match minmax {
        itertools::MinMaxResult::NoElements => None,
        itertools::MinMaxResult::OneElement(value) => {
            let scalar = make_scalar(dtype, value);
            Some(MinMaxResult {
                min: scalar.clone(),
                max: scalar,
            })
        }
        itertools::MinMaxResult::MinMax(min, max) => Some(MinMaxResult {
            min: make_scalar(dtype, min),
            max: make_scalar(dtype, max),
        }),
    })
}

fn make_scalar(dtype: &DType, value: &[u8]) -> Scalar {
    match dtype {
        DType::Binary(_) => Scalar::binary(value.to_vec(), NonNullable),
        DType::Utf8(_) => {
            // SAFETY: VarBin arrays always validate their data against their dtype.
            let value = unsafe { str::from_utf8_unchecked(value) };
            Scalar::utf8(value, NonNullable)
        }
        _ => vortex_panic!("cannot make Scalar from bytes with dtype {dtype}"),
    }
}

/// ZY-615 phase 0 measurement only, never merged: min, max, min/max and their bounded forms on a
/// VarBin array straight from its offsets and bytes, without executing it to views first.
///
/// The byte-wise extremes of the valid values go through the aggregate's own accumulator as a
/// two-value VarBinView array, so the partial is the one the canonical path produces: each of
/// these aggregates over a set depends only on the set's minimum and maximum.
#[derive(Debug)]
pub struct VarBinMinMaxKernel;

impl VarBinMinMaxKernel {
    /// The aggregates this kernel answers; register it for each on the VarBin encoding.
    pub fn aggregate_fn_ids() -> [crate::aggregate_fn::AggregateFnId; 5] {
        [
            MinMax.id(),
            Min.id(),
            Max.id(),
            BoundedMin.id(),
            BoundedMax.id(),
        ]
    }
}

impl DynAggregateKernel for VarBinMinMaxKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !Self::aggregate_fn_ids().contains(&aggregate_fn.id()) {
            return Ok(None);
        }
        let Some(varbin) = batch.as_opt::<VarBin>() else {
            return Ok(None);
        };
        let mask = varbin.validity()?.execute_mask(batch.len(), ctx)?;
        if matches!(mask.bit_buffer(), AllOr::None) {
            // All null (or empty): the canonical path has nothing to build either.
            return Ok(None);
        }
        let offsets = varbin.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let bytes = varbin.bytes().as_slice();
        let extremes = match_each_integer_ptype!(offsets.ptype(), |O| {
            offsets
                .as_slice::<O>()
                .windows(2)
                .enumerate()
                .filter(|(i, _)| mask.value(*i))
                .map(|(_, w)| {
                    let (start, end): (usize, usize) = (w[0].as_(), w[1].as_());
                    &bytes[start..end]
                })
                .minmax()
        });
        let (min, max) = match extremes {
            itertools::MinMaxResult::NoElements => return Ok(None),
            itertools::MinMaxResult::OneElement(value) => (value, value),
            itertools::MinMaxResult::MinMax(min, max) => (min, max),
        };
        let extremes =
            VarBinViewArray::from_iter([Some(min), Some(max)], batch.dtype().clone()).into_array();
        let mut accumulator = aggregate_fn.accumulator(batch.dtype())?;
        accumulator.accumulate(&extremes, ctx)?;
        accumulator.flush().map(Some)
    }
}
