// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::arrays_value_equal;
use super::is_constant;
use crate::ExecutionCtx;
use crate::arrays::ListViewArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::scalar_fn::fns::binary::nested_list_elements_are_constant;

/// Check if a list view array is constant by comparing each list's elements.
///
/// A list view array is constant if all lists have the same size and the same elements.
/// Nested elements reuse the existing null-safe comparator; other element types
/// retain `binary(Operator::Eq)` dispatch.
pub(super) fn check_listview_constant(
    l: &ListViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    if l.len() <= 1 {
        return Ok(true);
    }

    if !is_constant(l.sizes(), ctx)? {
        // If sizes aren't all equal, can't be constant.
        return Ok(false);
    }

    let first_size = l.size_at(0, ctx);
    if first_size == 0 {
        return Ok(true);
    }

    if is_constant(l.offsets(), ctx)? {
        // If all offsets are identical, every list references the same slice.
        return Ok(true);
    }

    // Preserve the original first-pair error and early-mismatch behavior before
    // preparing comparison state for the whole backing child.
    let first_elements = l.list_elements_at(0, ctx)?;
    let second_elements = l.list_elements_at(1, ctx)?;
    if !arrays_value_equal(&first_elements, &second_elements, ctx)? {
        return Ok(false);
    }
    if l.len() == 2 {
        return Ok(true);
    }

    if let Some(result) = nested_list_elements_are_constant(l, first_size, ctx)? {
        return Ok(result);
    }

    // Keep the original equality dispatch for other element types. The first pair
    // has already been checked on both paths.
    for i in 2..l.len() {
        let current_elements = l.list_elements_at(i, ctx)?;
        if !arrays_value_equal(&first_elements, &current_elements, ctx)? {
            return Ok(false);
        }
    }

    Ok(true)
}

#[cfg(test)]
#[path = "list_differential.rs"]
mod differential;
