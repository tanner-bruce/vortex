// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-wise comparison of nested (struct, list, fixed-size list, map) arrays.
//!
//! Nested comparisons canonicalize both operands recursively and build a tree of row
//! comparators, one per nested level. The ordering semantics match [`Scalar`] comparison:
//! structs compare field-by-field in declaration order, lists compare element-by-element and
//! then by length, maps compare as the list of their `{key, value}` entries, and a null value
//! (at any nesting depth) orders before every non-null value. Only top-level nulls make the
//! comparison result null.
//!
//! [`Scalar`]: crate::scalar::Scalar

use std::cmp::Ordering;

use num_traits::AsPrimitive;
use num_traits::ToPrimitive;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::List;
use crate::arrays::ListViewArray;
use crate::arrays::MapArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinViewArray;
use crate::arrays::decimal::widened_buffer;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::map::MapArraySlotsExt;
use crate::arrays::struct_::StructArrayExt;
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::operators::CompareOperator;

/// A row comparator: compares row `i` of the left operand against row `j` of the right operand.
type RowComparator = Box<dyn Fn(usize, usize) -> Ordering>;

/// Reuse this module's equality comparator for one nested-list constancy evaluation.
/// Other element types retain their existing encoded comparison dispatch.
pub(crate) fn nested_list_elements_are_constant(
    list: &ListViewArray,
    first_size: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<bool>> {
    let dtype = list.elements().dtype();
    if !matches!(
        dtype,
        DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..) | DType::Map(..)
    ) || !constancy_comparator_supported(dtype)
        || !constancy_backing_is_eager(list.elements())
        || !constancy_backing_is_eager(list.offsets())
    {
        return Ok(None);
    }

    // Borrow only already-primitive host offsets. Executing an encoding here could
    // evaluate offsets the original per-pair path would never consume.
    let Some(offsets) = list.offsets().as_opt::<Primitive>() else {
        return Ok(None);
    };

    // The caller has checked equal, nonzero sizes, unequal offsets and the first pair.
    // The canonical backing and comparator live only for this statistic evaluation. Prepare them once,
    // rather than recursively executing and comparing two fresh slices for every list.
    let elements = list
        .elements()
        .clone()
        .execute::<RecursiveCanonical>(ctx)?
        .0
        .into_array();
    let comparator = build_comparator(&elements, &elements, ctx)?;
    match_each_integer_ptype!(offsets.ptype(), |P| {
        let offsets = offsets.as_slice::<P>();
        let first: usize = offsets[0].as_();
        for &offset in &offsets[2..] {
            let offset: usize = offset.as_();
            for element in 0..first_size {
                if !comparator(first + element, offset + element).is_eq() {
                    return Ok(Some(false));
                }
            }
        }
        Ok(Some(true))
    })
}

// Whole-child preparation must not evaluate lazy expressions in unused ranges, nor
// introduce device execution that the original validity/early-exit path could avoid.
// Canonical arrays, Arrow-style lists and offset-based byte arrays have materialized
// backing. Inspect every stored child slot, including validity, without executing
// unknown encodings or allocating payloads. List constructors validate their offsets;
// VarBin's public binary constructor does not, so prove its conversion bounds here.
fn constancy_backing_is_eager(array: &ArrayRef) -> bool {
    if !(array.is::<AnyCanonical>() || array.is::<List>() || array.is::<VarBin>())
        || !array.buffer_handles().iter().all(BufferHandle::is_on_host)
        || !array.children_iter().all(constancy_backing_is_eager)
    {
        return false;
    }
    let Some(varbin) = array.as_opt::<VarBin>() else {
        return true;
    };
    let Some(offsets) = varbin.offsets().as_opt::<Primitive>() else {
        return false;
    };
    if !offsets.dtype().is_int()
        || offsets.dtype().is_nullable()
        || offsets.len().checked_sub(1) != Some(array.len())
    {
        return false;
    }
    let bytes_len = varbin.bytes_handle().len();
    match_each_integer_ptype!(offsets.ptype(), |P| {
        let mut previous = 0usize;
        for (index, &offset) in offsets.as_slice::<P>().iter().enumerate() {
            let Some(offset) = offset.to_usize() else {
                return false;
            };
            if offset > bytes_len
                || (index != 0 && (offset < previous || offset - previous > MAX_BUFFER_LEN))
            {
                return false;
            }
            previous = offset;
        }
        true
    })
}

// Eager setup must not introduce unsupported-child errors that the original per-pair
// validity/early-mismatch checks could avoid. Keep those dtypes on the original path.
fn constancy_comparator_supported(dtype: &DType) -> bool {
    match dtype {
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
        DType::Struct(fields, _) => fields
            .fields()
            .all(|dtype| constancy_comparator_supported(&dtype)),
        DType::List(element, _) | DType::FixedSizeList(element, ..) => {
            constancy_comparator_supported(element)
        }
        DType::Map(map, _) => {
            constancy_comparator_supported(&map.key_dtype())
                && constancy_comparator_supported(&map.value_dtype())
        }
        DType::Extension(extension) => constancy_comparator_supported(extension.storage_dtype()),
        DType::Union(..) | DType::Variant(_) => false,
    }
}

/// Compare two nested arrays row by row.
pub(super) fn compare_nested(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let lhs = lhs
        .clone()
        .execute::<RecursiveCanonical>(ctx)?
        .0
        .into_array();
    let rhs = rhs
        .clone()
        .execute::<RecursiveCanonical>(ctx)?
        .0
        .into_array();

    let validity = compare_validity(lhs.validity()?, rhs.validity()?, nullability)?;
    let comparator = build_comparator(&lhs, &rhs, ctx)?;
    // Dispatch the operator outside the row loop so the predicate inlines into each loop; the
    // comparator call itself stays virtual.
    let bits = match op {
        CompareOperator::Eq => collect_ordering_bits(len, &comparator, Ordering::is_eq),
        CompareOperator::NotEq => collect_ordering_bits(len, &comparator, Ordering::is_ne),
        CompareOperator::Gt => collect_ordering_bits(len, &comparator, Ordering::is_gt),
        CompareOperator::Gte => collect_ordering_bits(len, &comparator, Ordering::is_ge),
        CompareOperator::Lt => collect_ordering_bits(len, &comparator, Ordering::is_lt),
        CompareOperator::Lte => collect_ordering_bits(len, &comparator, Ordering::is_le),
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

/// Bit-pack `predicate(comparator(i, i))` over `len` rows.
fn collect_ordering_bits(
    len: usize,
    comparator: &RowComparator,
    predicate: impl Fn(Ordering) -> bool,
) -> BitBuffer {
    BitBuffer::collect_bool(len, |i| predicate(comparator(i, i)))
}

/// The validity mask of a recursively canonical array.
fn validity_mask(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
    array.validity()?.execute_mask(array.len(), ctx)
}

/// Build a row comparator over two recursively canonical arrays of the same logical dtype
/// (ignoring nullability). Null values order before all non-null values at every level.
fn build_comparator(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RowComparator> {
    let lhs_mask = validity_mask(lhs, ctx)?;
    let rhs_mask = validity_mask(rhs, ctx)?;
    let values = build_values_comparator(lhs, rhs, ctx)?;

    if lhs_mask.all_true() && rhs_mask.all_true() {
        return Ok(values);
    }

    Ok(Box::new(move |i, j| {
        match (lhs_mask.value(i), rhs_mask.value(j)) {
            (true, true) => values(i, j),
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => Ordering::Equal,
        }
    }))
}

/// Build a comparator over the non-null values of two recursively canonical arrays.
fn build_values_comparator(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RowComparator> {
    if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
        vortex_bail!(
            "Cannot compare different DTypes {} and {}",
            lhs.dtype(),
            rhs.dtype()
        );
    }

    Ok(match lhs.dtype() {
        // Null values compare through the validity wrapper; the value comparator is trivial.
        DType::Null => Box::new(|_, _| Ordering::Equal),
        DType::Bool(_) => {
            let lhs = lhs.clone().execute::<BoolArray>(ctx)?.into_bit_buffer();
            let rhs = rhs.clone().execute::<BoolArray>(ctx)?.into_bit_buffer();
            Box::new(move |i, j| lhs.value(i).cmp(&rhs.value(j)))
        }
        DType::Primitive(ptype, _) => {
            match_each_native_ptype!(*ptype, |T| { primitive_comparator::<T>(lhs, rhs, ctx)? })
        }
        DType::Decimal(..) => {
            let lhs = lhs.clone().execute::<DecimalArray>(ctx)?;
            let rhs = rhs.clone().execute::<DecimalArray>(ctx)?;
            let common = lhs.values_type().max(rhs.values_type());
            crate::match_each_decimal_value_type!(common, |W| {
                let lhs = widened_buffer::<W>(&lhs);
                let rhs = widened_buffer::<W>(&rhs);
                Box::new(move |i: usize, j: usize| lhs[i].cmp(&rhs[j])) as RowComparator
            })
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let lhs = lhs.clone().execute::<VarBinViewArray>(ctx)?;
            let rhs = rhs.clone().execute::<VarBinViewArray>(ctx)?;
            Box::new(move |i, j| view_bytes(&lhs, i).cmp(view_bytes(&rhs, j)))
        }
        DType::Struct(..) => {
            let lhs = lhs.clone().execute::<StructArray>(ctx)?;
            let rhs = rhs.clone().execute::<StructArray>(ctx)?;
            let fields = lhs
                .iter_unmasked_fields()
                .zip(rhs.iter_unmasked_fields())
                .map(|(lhs_field, rhs_field)| build_comparator(lhs_field, rhs_field, ctx))
                .collect::<VortexResult<Vec<_>>>()?;
            Box::new(move |i, j| {
                fields
                    .iter()
                    .map(|cmp| cmp(i, j))
                    .find(|ordering| ordering.is_ne())
                    .unwrap_or(Ordering::Equal)
            })
        }
        DType::List(..) => {
            let same_array = ArrayRef::ptr_eq(lhs, rhs);
            let lhs = lhs.clone().execute::<ListViewArray>(ctx)?;
            let rhs = rhs.clone().execute::<ListViewArray>(ctx)?;
            let elements = build_comparator(lhs.elements(), rhs.elements(), ctx)?;
            // Materialize offsets and sizes once instead of paying `offset_at`/`size_at`'s
            // per-call encoding dispatch inside the row loop.
            let lhs_offsets = usize_values(lhs.offsets(), ctx)?;
            let lhs_sizes = usize_values(lhs.sizes(), ctx)?;
            if same_array {
                return Ok(Box::new(move |i, j| {
                    let lhs_len = lhs_sizes[i];
                    let rhs_len = lhs_sizes[j];
                    (0..lhs_len.min(rhs_len))
                        .map(|el| elements(lhs_offsets[i] + el, lhs_offsets[j] + el))
                        .find(|ordering| ordering.is_ne())
                        .unwrap_or_else(|| lhs_len.cmp(&rhs_len))
                }));
            }
            let rhs_offsets = usize_values(rhs.offsets(), ctx)?;
            let rhs_sizes = usize_values(rhs.sizes(), ctx)?;
            Box::new(move |i, j| {
                let lhs_len = lhs_sizes[i];
                let rhs_len = rhs_sizes[j];
                (0..lhs_len.min(rhs_len))
                    .map(|el| elements(lhs_offsets[i] + el, rhs_offsets[j] + el))
                    .find(|ordering| ordering.is_ne())
                    .unwrap_or_else(|| lhs_len.cmp(&rhs_len))
            })
        }
        DType::FixedSizeList(_, size, _) => {
            let size = *size as usize;
            let lhs = lhs.clone().execute::<FixedSizeListArray>(ctx)?;
            let rhs = rhs.clone().execute::<FixedSizeListArray>(ctx)?;
            let elements = build_comparator(lhs.elements(), rhs.elements(), ctx)?;
            Box::new(move |i, j| {
                (0..size)
                    .map(|el| elements(i * size + el, j * size + el))
                    .find(|ordering| ordering.is_ne())
                    .unwrap_or(Ordering::Equal)
            })
        }
        DType::Extension(_) => {
            let lhs = lhs.clone().execute::<ExtensionArray>(ctx)?;
            let rhs = rhs.clone().execute::<ExtensionArray>(ctx)?;
            build_comparator(lhs.storage_array(), rhs.storage_array(), ctx)?
        }
        DType::Map(..) => {
            let lhs = lhs.clone().execute::<MapArray>(ctx)?;
            let rhs = rhs.clone().execute::<MapArray>(ctx)?;
            build_values_comparator(lhs.entries(), rhs.entries(), ctx)?
        }
        DType::Union(..) | DType::Variant(_) => {
            vortex_bail!("compare is not supported for dtype {}", lhs.dtype())
        }
    })
}

fn primitive_comparator<T: NativePType>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RowComparator> {
    let lhs = lhs
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<T>();
    let rhs = rhs
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .into_buffer::<T>();
    Ok(Box::new(move |i, j| lhs[i].total_compare(rhs[j])))
}

/// Materialize an integer array as `usize` values for `O(1)` row access.
fn usize_values(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<usize>> {
    let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
    match_each_integer_ptype!(primitive.ptype(), |P| {
        Ok(primitive.as_slice::<P>().iter().map(|v| v.as_()).collect())
    })
}

/// The bytes of row `index`, resolved directly from the view without cloning buffer handles the
/// way `VarBinViewArray::bytes_at` does.
fn view_bytes(array: &VarBinViewArray, index: usize) -> &[u8] {
    let view = &array.views()[index];
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view = view.as_view();
        &array.buffer(view.buffer_index as usize).as_slice()[view.as_range()]
    }
}

#[cfg(test)]
#[path = "nested_tests.rs"]
mod tests;
