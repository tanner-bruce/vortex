// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Differential oracle retained from vortex-array 0.86.1's original list loop.
use half::f16;
use num_traits::AsPrimitive;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::super::arrays_value_equal;
use super::super::is_constant;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::MapArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::UnionArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::VariantArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::FieldNames;
use crate::dtype::MapDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::UnionVariants;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::expr::stats::StatsProviderExt;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::nested_list_elements_are_constant;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

// Exact original check_listview_constant body: this oracle intentionally retains per-pair
// slices and the original null-safe equality dispatch instead of sharing the new helper.
fn original_list(l: &ListViewArray, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    if l.len() <= 1 {
        return Ok(true);
    }
    if !is_constant(l.sizes(), ctx)? {
        return Ok(false);
    }
    let first_size = l.size_at(0);
    if first_size == 0 {
        return Ok(true);
    }
    if is_constant(l.offsets(), ctx)? {
        return Ok(true);
    }
    let first_elements = l.list_elements_at(0)?;
    for i in 1..l.len() {
        let current_elements = l.list_elements_at(i)?;
        if !arrays_value_equal(&first_elements, &current_elements, ctx)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn view(elements: ArrayRef, offsets: &[u64], sizes: &[u64], validity: Validity) -> ListViewArray {
    ListViewArray::new(
        elements,
        PrimitiveArray::from_iter(offsets.iter().copied()).into_array(),
        PrimitiveArray::from_iter(sizes.iter().copied()).into_array(),
        validity,
    )
}
fn structure(child: ArrayRef, validity: Validity) -> VortexResult<ArrayRef> {
    Ok(StructArray::try_new(
        FieldNames::from(["x"]),
        vec![child.clone()],
        child.len(),
        validity,
    )?
    .into_array())
}
fn outer(child: ArrayRef) -> ListViewArray {
    view(child, &[0, 2, 4], &[2, 2, 2], Validity::NonNullable)
}

fn compare(
    name: &str,
    build: impl Fn() -> VortexResult<ListViewArray>,
    expected: Option<bool>,
) -> VortexResult<()> {
    // Fresh independent construction avoids cached statistics from the oracle warming the candidate.
    let old = build()?;
    let new = build()?;
    let mut old_ctx = array_session().create_execution_ctx();
    let mut new_ctx = array_session().create_execution_ctx();
    let old_result = if old.is_empty() {
        Ok(false)
    } else {
        let mask = old.validity()?.execute_mask(old.len(), &mut old_ctx)?;
        if mask.all_false() {
            Ok(true)
        } else if !mask.all_true() {
            Ok(false)
        } else {
            original_list(&old, &mut old_ctx)
        }
    };
    let new = new.into_array();
    let new_result = is_constant(&new, &mut new_ctx);
    match (old_result, new_result) {
        (Ok(old_value), Ok(new_value)) => {
            assert_eq!(new_value, old_value, "{name}: value");
            if let Some(expected) = expected {
                assert_eq!(new_value, expected, "{name}: known outcome");
            }
            let old = old.into_array();
            if !old.is_empty() {
                old.statistics()
                    .set(Stat::IsConstant, Precision::Exact(old_value.into()));
            }
            assert_eq!(
                new.statistics().get(Stat::IsConstant),
                old.statistics().get(Stat::IsConstant),
                "{name}: value/precision"
            );
            old.statistics()
                .compute_all(&[Stat::NullCount], &mut old_ctx)?;
            new.statistics()
                .compute_all(&[Stat::NullCount], &mut new_ctx)?;
            assert_eq!(
                new.statistics().get(Stat::NullCount),
                old.statistics().get(Stat::NullCount),
                "{name}: null count/precision"
            );
            assert_eq!(
                is_constant(&new, &mut new_ctx)?,
                old_value,
                "{name}: cached call"
            );
        }
        (Err(old_error), Err(new_error)) => {
            assert_eq!(
                new_error.to_string(),
                old_error.to_string(),
                "{name}: original error"
            );
            assert!(
                expected.is_none(),
                "{name}: unexpectedly failed known outcome"
            );
            assert_eq!(
                new.statistics().get_as::<bool>(Stat::IsConstant),
                Precision::Absent,
                "{name}: failed computation cached no value"
            );
        }
        (old_result, new_result) => {
            return Err(vortex_err!(
                "{name}: oracle {old_result:?}, candidate {new_result:?}"
            ));
        }
    }
    Ok(())
}

#[test]
fn every_primitive_type_flat_control_and_nested_dispatch() -> VortexResult<()> {
    macro_rules! primitive {
        ($ty:ty) => {
            for nested in [false, true] {
                for mismatch in [None, Some(0), Some(5)] {
                    compare(
                        concat!(stringify!($ty), " numeric"),
                        || {
                            let mut values =
                                [1 as $ty, 2 as $ty, 1 as $ty, 2 as $ty, 1 as $ty, 2 as $ty];
                            if let Some(index) = mismatch {
                                values[index] = 3 as $ty;
                            }
                            let child = PrimitiveArray::from_iter(values).into_array();
                            Ok(outer(if nested {
                                structure(child, Validity::NonNullable)?
                            } else {
                                child
                            }))
                        },
                        Some(mismatch.is_none()),
                    )?;
                }
            }
        };
    }
    primitive!(u8);
    primitive!(u16);
    primitive!(u32);
    primitive!(u64);
    primitive!(i8);
    primitive!(i16);
    primitive!(i32);
    primitive!(i64);
    primitive!(f32);
    primitive!(f64);
    compare(
        "f16",
        || {
            Ok(outer(structure(
                PrimitiveArray::from_iter([1., 2., 1., 2., 1., 2.].map(f16::from_f32)).into_array(),
                Validity::NonNullable,
            )?))
        },
        Some(true),
    )?;
    Ok(())
}

#[test]
fn nan_payloads_signed_zero_and_infinity_preserve_each_dispatch() -> VortexResult<()> {
    let nan_a = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan_b = f64::from_bits(0x7ff8_0000_0000_0002);
    for values in [
        [nan_a, 1., nan_a, 1., nan_a, 1.],
        [nan_a, 1., nan_b, 1., nan_a, 1.],
        [0., 1., -0., 1., 0., 1.],
        [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ],
    ] {
        for nested in [false, true] {
            compare(
                "float semantics",
                || {
                    let child = PrimitiveArray::from_iter(values).into_array();
                    Ok(outer(if nested {
                        structure(child, Validity::NonNullable)?
                    } else {
                        child
                    }))
                },
                None,
            )?;
        }
    }
    Ok(())
}

#[test]
fn bool_decimal_utf8_and_binary_controls() -> VortexResult<()> {
    for nested in [false, true] {
        for changed in [false, true] {
            for kind in 0..4 {
                compare(
                    "leaf controls",
                    || {
                        let child = match kind {
                            0 => BoolArray::from_iter([true, false, true, false, true, changed])
                                .into_array(),
                            1 => DecimalArray::from_iter(
                                [100i128, 200, 100, 200, 100, if changed { 300 } else { 200 }],
                                DecimalDType::new(19, 2),
                            )
                            .into_array(),
                            2 => VarBinViewArray::from_iter_str([
                                "short",
                                "long non-inline string value",
                                "short",
                                "long non-inline string value",
                                "short",
                                if changed {
                                    "different"
                                } else {
                                    "long non-inline string value"
                                },
                            ])
                            .into_array(),
                            _ => VarBinViewArray::from_iter_bin([
                                b"\0\xff".as_slice(),
                                b"abcdefghijklmnop",
                                b"\0\xff",
                                b"abcdefghijklmnop",
                                b"\0\xff",
                                if changed {
                                    b"different"
                                } else {
                                    b"abcdefghijklmnop"
                                },
                            ])
                            .into_array(),
                        };
                        Ok(outer(if nested {
                            structure(child, Validity::NonNullable)?
                        } else {
                            child
                        }))
                    },
                    Some(!changed),
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn nested_list_fixed_size_map_and_extension_trees() -> VortexResult<()> {
    for changed in [false, true] {
        for kind in 0..5 {
            compare(
                "nested tree",
                || {
                    let mut values = [1i64, 2].repeat(6);
                    if changed {
                        values[11] = 3;
                    }
                    let leaf = PrimitiveArray::from_iter(values).into_array();
                    let child = match kind {
                        0 => view(leaf, &[0, 2, 4, 6, 8, 10], &[2; 6], Validity::NonNullable)
                            .into_array(),
                        1 => FixedSizeListArray::try_new(leaf, 2, Validity::NonNullable, 6)?
                            .into_array(),
                        2 => {
                            let entries = StructArray::try_new(
                                FieldNames::from(["key", "value"]),
                                vec![
                                    PrimitiveArray::from_iter([1i32, 2].repeat(6)).into_array(),
                                    leaf,
                                ],
                                12,
                                Validity::NonNullable,
                            )?;
                            let dtype = MapDType::try_new(
                                DType::Primitive(PType::I32, Nullability::NonNullable),
                                DType::Primitive(PType::I64, Nullability::NonNullable),
                                false,
                            )?;
                            MapArray::try_new(
                                dtype,
                                view(
                                    entries.into_array(),
                                    &[0, 2, 4, 6, 8, 10],
                                    &[2; 6],
                                    Validity::NonNullable,
                                ),
                            )?
                            .into_array()
                        }
                        3 => {
                            let storage = PrimitiveArray::from_iter([
                                1i64,
                                2,
                                1,
                                2,
                                1,
                                if changed { 3 } else { 2 },
                            ])
                            .into_array();
                            structure(
                                ExtensionArray::try_new(
                                    Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable)
                                        .erased(),
                                    storage,
                                )?
                                .into_array(),
                                Validity::NonNullable,
                            )?
                        }
                        _ => structure(
                            view(leaf, &[0, 2, 4, 6, 8, 10], &[2; 6], Validity::NonNullable)
                                .into_array(),
                            Validity::NonNullable,
                        )?,
                    };
                    Ok(outer(child))
                },
                Some(!changed),
            )?;
        }
    }
    Ok(())
}

#[test]
fn parent_nested_and_leaf_nulls() -> VortexResult<()> {
    for parent in [
        Validity::NonNullable,
        Validity::AllInvalid,
        Validity::from_iter([true, false, true]),
    ] {
        compare(
            "parent nulls",
            || {
                Ok(view(
                    structure(
                        PrimitiveArray::from_option_iter([
                            Some(1i32),
                            None,
                            Some(1),
                            None,
                            Some(1),
                            None,
                        ])
                        .into_array(),
                        Validity::NonNullable,
                    )?,
                    &[0, 2, 4],
                    &[2; 3],
                    parent.clone(),
                ))
            },
            Some(!matches!(parent, Validity::Array(_))),
        )?;
    }
    compare(
        "null struct hides unequal backing",
        || {
            Ok(outer(structure(
                PrimitiveArray::from_iter([1i32, 99, 1, 98, 1, 97]).into_array(),
                Validity::from_iter([true, false, true, false, true, false]),
            )?))
        },
        Some(true),
    )?;
    compare(
        "different nested mask",
        || {
            Ok(outer(structure(
                PrimitiveArray::from_iter([1i32, 2, 1, 2, 1, 2]).into_array(),
                Validity::from_iter([true, false, true, true, true, false]),
            )?))
        },
        Some(false),
    )?;
    compare(
        "all null leaves",
        || {
            Ok(outer(structure(
                NullArray::new(6).into_array(),
                Validity::NonNullable,
            )?))
        },
        Some(true),
    )?;
    compare(
        "all null structs",
        || {
            Ok(outer(structure(
                PrimitiveArray::from_iter([1i32, 2, 9, 8, 7, 6]).into_array(),
                Validity::AllInvalid,
            )?))
        },
        Some(true),
    )?;
    Ok(())
}

#[test]
fn empty_single_unequal_sizes_repeated_offsets_and_unused_children() -> VortexResult<()> {
    for (offsets, sizes, expected) in [
        (vec![], vec![], false),
        (vec![1], vec![2], true),
        (vec![0, 1, 2], vec![0, 0, 0], true),
        (vec![0, 1], vec![1, 2], false),
        (vec![2, 2, 2], vec![2, 2, 2], true),
        (vec![0, 4, 8], vec![2, 2, 2], true),
    ] {
        compare(
            "fast path / gapped unused",
            || {
                Ok(view(
                    structure(
                        PrimitiveArray::from_iter([1i32, 2, 99, 98, 1, 2, 97, 96, 1, 2])
                            .into_array(),
                        Validity::NonNullable,
                    )?,
                    &offsets,
                    &sizes,
                    Validity::NonNullable,
                ))
            },
            Some(expected),
        )?;
    }
    compare(
        "sliced outer",
        || {
            let mut ctx = array_session().create_execution_ctx();
            view(
                structure(
                    PrimitiveArray::from_iter([9i32, 9, 1, 2, 1, 2, 1, 2]).into_array(),
                    Validity::NonNullable,
                )?,
                &[0, 2, 4, 6],
                &[2; 4],
                Validity::NonNullable,
            )
            .slice(1..4)?
            .execute::<ListViewArray>(&mut ctx)
        },
        Some(true),
    )?;
    // Enumerate valid out-of-order, repeated and overlapping ranges independently of the candidate.
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for size in 1..=2 {
                    compare(
                        "exhaustive small views",
                        || {
                            Ok(view(
                                structure(
                                    PrimitiveArray::from_iter([1i32, 2, 1, 2, 9, 8]).into_array(),
                                    Validity::NonNullable,
                                )?,
                                &[a, b, c],
                                &[size; 3],
                                Validity::NonNullable,
                            ))
                        },
                        None,
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn unsupported_nested_union_preserves_errors_and_validity_short_circuits() -> VortexResult<()> {
    for mask in [
        Validity::NonNullable,
        Validity::AllInvalid,
        Validity::from_iter([true, false, false, true, true, false]),
    ] {
        compare(
            "unsupported nested union",
            || {
                let variants = UnionVariants::try_new(
                    ["number"].into(),
                    vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
                    vec![5],
                )?;
                let union = UnionArray::try_new(
                    PrimitiveArray::from_iter([5u8; 6]).into_array(),
                    variants,
                    vec![PrimitiveArray::from_iter([1i32, 2, 1, 2, 1, 2]).into_array()],
                )?;
                Ok(outer(structure(union.into_array(), mask.clone())?))
            },
            None,
        )?;
    }
    for mask in [
        Validity::NonNullable,
        Validity::AllInvalid,
        Validity::from_iter([true, false, false, true, true, false]),
    ] {
        compare(
            "unsupported nested variant",
            || {
                let core = ConstantArray::new(
                    Scalar::variant(Scalar::primitive(1i32, Nullability::NonNullable)),
                    6,
                )
                .into_array();
                let variant = VariantArray::try_new(core, None)?;
                Ok(outer(structure(variant.into_array(), mask.clone())?))
            },
            None,
        )?;
    }
    Ok(())
}

#[test]
fn chunked_sliced_and_nullable_string_children() -> VortexResult<()> {
    compare(
        "chunked child",
        || {
            let child = ChunkedArray::try_new(
                vec![
                    PrimitiveArray::from_iter([1i32, 2, 1]).into_array(),
                    PrimitiveArray::from_iter([2i32, 1, 2]).into_array(),
                ],
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )?
            .into_array();
            Ok(outer(structure(child, Validity::NonNullable)?))
        },
        Some(true),
    )?;
    compare(
        "sliced child",
        || {
            Ok(outer(structure(
                PrimitiveArray::from_iter([9i32, 1, 2, 1, 2, 1, 2, 8])
                    .into_array()
                    .slice(1..7)?,
                Validity::NonNullable,
            )?))
        },
        Some(true),
    )?;
    for different in [false, true] {
        compare(
            "nullable strings",
            || {
                Ok(outer(structure(
                    VarBinViewArray::from_iter_nullable_str([
                        Some("long string beyond inline"),
                        None,
                        Some("long string beyond inline"),
                        None,
                        Some("long string beyond inline"),
                        different.then_some("changed"),
                    ])
                    .into_array(),
                    Validity::NonNullable,
                )?))
            },
            Some(!different),
        )?;
    }
    compare(
        "nested varied lengths",
        || {
            let inner = view(
                PrimitiveArray::from_iter([1i32, 2, 3, 1, 2, 3]).into_array(),
                &[0, 0, 1, 3, 3, 4],
                &[0, 1, 2, 0, 1, 2],
                Validity::NonNullable,
            );
            Ok(view(
                inner.into_array(),
                &[0, 3],
                &[3, 3],
                Validity::NonNullable,
            ))
        },
        Some(true),
    )?;
    Ok(())
}

#[test]
fn f32_and_f16_special_value_controls() -> VortexResult<()> {
    for nested in [false, true] {
        for values in [
            [
                f32::from_bits(0x7fc0_0001),
                1.,
                f32::from_bits(0x7fc0_0001),
                1.,
            ],
            [
                f32::from_bits(0x7fc0_0001),
                1.,
                f32::from_bits(0x7fc0_0002),
                1.,
            ],
            [0., 1., -0., 1.],
        ] {
            compare(
                "f32 special",
                || {
                    let child = PrimitiveArray::from_iter(values).into_array();
                    Ok(view(
                        if nested {
                            structure(child, Validity::NonNullable)?
                        } else {
                            child
                        },
                        &[0, 2],
                        &[2; 2],
                        Validity::NonNullable,
                    ))
                },
                None,
            )?;
        }
        let one = f16::from_f32(1.);
        for values in [
            [f16::from_bits(0x7e01), one, f16::from_bits(0x7e01), one],
            [f16::from_bits(0x7e01), one, f16::from_bits(0x7e02), one],
            [f16::from_f32(0.), one, f16::from_f32(-0.), one],
        ] {
            compare(
                "f16 special",
                || {
                    let child = PrimitiveArray::from_iter(values).into_array();
                    Ok(view(
                        if nested {
                            structure(child, Validity::NonNullable)?
                        } else {
                            child
                        },
                        &[0, 2],
                        &[2; 2],
                        Validity::NonNullable,
                    ))
                },
                None,
            )?;
        }
    }
    Ok(())
}

#[test]
fn supported_lazy_error_in_unused_child_range() -> VortexResult<()> {
    compare(
        "unused child division error",
        || {
            let values = PrimitiveArray::from_iter([1i32, 2, 1, 2, 9]).into_array();
            let divisors = PrimitiveArray::from_iter([1i32, 1, 1, 1, 0]).into_array();
            let lazy = values.binary(divisors, Operator::Div)?;
            Ok(view(
                structure(lazy, Validity::NonNullable)?,
                &[0, 2],
                &[2; 2],
                Validity::NonNullable,
            ))
        },
        Some(true),
    )
}

#[test]
fn supported_lazy_error_after_early_mismatch() -> VortexResult<()> {
    compare(
        "early mismatch before division error",
        || {
            let values = PrimitiveArray::from_iter([1i32, 2, 3, 4, 9]).into_array();
            let divisors = PrimitiveArray::from_iter([1i32, 1, 1, 1, 0]).into_array();
            let lazy = values.binary(divisors, Operator::Div)?;
            Ok(view(
                structure(lazy, Validity::NonNullable)?,
                &[0, 2],
                &[2; 2],
                Validity::NonNullable,
            ))
        },
        Some(false),
    )
}

fn lazy_validity_view() -> VortexResult<ListViewArray> {
    let numerator = PrimitiveArray::from_iter([1i32; 5]).into_array();
    let divisor = PrimitiveArray::from_iter([1i32, 1, 1, 1, 0]).into_array();
    let valid = numerator
        .binary(divisor, Operator::Div)?
        .binary(ConstantArray::new(1i32, 5).into_array(), Operator::Eq)?;
    let values =
        PrimitiveArray::new(buffer![1i32, 2, 1, 2, 9], Validity::Array(valid)).into_array();
    Ok(view(
        structure(values, Validity::NonNullable)?,
        &[0, 2],
        &[2; 2],
        Validity::NonNullable,
    ))
}

#[test]
fn eager_guard_preserves_encoded_and_validity_fallback() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let eager = outer(structure(
        PrimitiveArray::from_iter([1i32, 2, 1, 2, 1, 2]).into_array(),
        Validity::NonNullable,
    )?);
    assert_eq!(
        nested_list_elements_are_constant(&eager, 2, &mut ctx)?,
        Some(true)
    );
    let encoded = ChunkedArray::try_new(
        vec![
            PrimitiveArray::from_iter([1i32, 2]).into_array(),
            PrimitiveArray::from_iter([1i32, 2]).into_array(),
        ],
        DType::Primitive(PType::I32, Nullability::NonNullable),
    )?
    .into_array();
    let encoded = view(
        structure(encoded, Validity::NonNullable)?,
        &[0, 2],
        &[2; 2],
        Validity::NonNullable,
    );
    assert_eq!(
        nested_list_elements_are_constant(&encoded, 2, &mut ctx)?,
        None
    );
    // The guard must inspect the lazy validity descendant without evaluating its unused error.
    assert_eq!(
        nested_list_elements_are_constant(&lazy_validity_view()?, 2, &mut ctx)?,
        None
    );
    compare(
        "unused validity division error",
        lazy_validity_view,
        Some(true),
    )?;
    let referenced_error = || {
        let values = PrimitiveArray::from_iter([1i32, 2, 1, 2]).into_array();
        let divisors = PrimitiveArray::from_iter([1i32, 1, 1, 0]).into_array();
        let lazy = values.binary(divisors, Operator::Div)?;
        Ok(view(
            structure(lazy, Validity::NonNullable)?,
            &[0, 2],
            &[2; 2],
            Validity::NonNullable,
        ))
    };
    assert!(original_list(&referenced_error()?, &mut ctx).is_err());
    compare(
        "referenced division error preserved",
        referenced_error,
        None,
    )?;
    Ok(())
}

#[test]
fn eager_guard_preserves_encoded_outer_offsets_fallback() -> VortexResult<()> {
    for different in [false, true] {
        let build = || {
            let offsets = ChunkedArray::try_new(
                vec![
                    PrimitiveArray::from_iter([0u64, 2]).into_array(),
                    PrimitiveArray::from_iter([4u64]).into_array(),
                ],
                DType::Primitive(PType::U64, Nullability::NonNullable),
            )?
            .into_array();
            // A cached false statistic can avoid executing the entire offsets array
            // on the original path; the reusable helper must retain its fallback.
            offsets
                .statistics()
                .set(Stat::IsConstant, Precision::Exact(false.into()));
            let values = [1i32, 2, 1, 2, if different { 3 } else { 1 }, 2];
            ListViewArray::try_new(
                structure(
                    PrimitiveArray::from_iter(values).into_array(),
                    Validity::NonNullable,
                )?,
                offsets,
                PrimitiveArray::from_iter([2u64; 3]).into_array(),
                Validity::NonNullable,
            )
        };
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(
            nested_list_elements_are_constant(&build()?, 2, &mut ctx)?,
            None
        );
        compare("encoded outer offsets", build, Some(!different))?;
    }
    Ok(())
}

#[test]
fn materialized_arrow_list_and_varbin_dispatch() -> VortexResult<()> {
    for different in [false, true] {
        let build = || {
            let mut values = vec![1i32, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
            if different {
                values[8] = 9;
            }
            let inner = ListArray::try_new(
                PrimitiveArray::from_iter(values).into_array(),
                PrimitiveArray::from_iter([0i32, 2, 4, 6, 8, 10, 12]).into_array(),
                Validity::NonNullable,
            )?
            .into_array();
            Ok(outer(structure(inner, Validity::NonNullable)?))
        };
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(
            nested_list_elements_are_constant(&build()?, 2, &mut ctx)?,
            Some(!different)
        );
        compare("materialized Arrow list", build, Some(!different))?;
        for dtype in [
            DType::Utf8(Nullability::NonNullable),
            DType::Binary(Nullability::NonNullable),
        ] {
            macro_rules! offsets_control {
                ($ptype:ty) => {{
                    let build = || {
                        let mut bytes = b"aabbaabbaabb".to_vec();
                        if different {
                            bytes[8] = b'c';
                        }
                        let child = VarBinArray::try_new(
                            PrimitiveArray::from_iter([0 as $ptype, 2, 4, 6, 8, 10, 12])
                                .into_array(),
                            bytes.into(),
                            dtype.clone(),
                            Validity::NonNullable,
                        )?
                        .into_array();
                        Ok(outer(structure(child, Validity::NonNullable)?))
                    };
                    let mut ctx = array_session().create_execution_ctx();
                    assert_eq!(
                        nested_list_elements_are_constant(&build()?, 2, &mut ctx)?,
                        Some(!different)
                    );
                    compare("materialized VarBin offsets", build, Some(!different))?;
                }};
            }
            offsets_control!(i32);
            offsets_control!(i64);
            offsets_control!(u32);
            offsets_control!(u64);
        }
    }
    Ok(())
}

#[test]
fn malformed_unused_binary_offsets_preserve_fallback() -> VortexResult<()> {
    for last in [-1i32, 11, 99] {
        let build = || {
            // The public Binary constructor accepts these malformed unused tails.
            // Whole-child canonicalization would panic; referenced prefixes are valid.
            let child = VarBinArray::try_new(
                PrimitiveArray::from_iter([0i32, 2, 4, 6, 8, 10, 12, last]).into_array(),
                b"aabbaabbaabb".to_vec().into(),
                DType::Binary(Nullability::NonNullable),
                Validity::NonNullable,
            )?
            .into_array();
            Ok(outer(structure(child, Validity::NonNullable)?))
        };
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(
            nested_list_elements_are_constant(&build()?, 2, &mut ctx)?,
            None
        );
        compare("malformed unused binary offset", build, Some(true))?;
    }
    Ok(())
}

fn lazy_varbin_offsets(error_index: usize, different: bool) -> VortexResult<ListViewArray> {
    let numerator = PrimitiveArray::from_iter([0i32, 2, 4, 6, 8, 10, 12, 14]).into_array();
    let mut divisors = vec![1i32; 8];
    divisors[error_index] = 0;
    let offsets = numerator.binary(
        PrimitiveArray::from_iter(divisors).into_array(),
        Operator::Div,
    )?;
    let mut bytes = b"aabbaabbaabbzz".to_vec();
    if different {
        bytes[4] = b'c';
    }
    let child = VarBinArray::try_new(
        offsets,
        bytes.into(),
        DType::Binary(Nullability::NonNullable),
        Validity::NonNullable,
    )?
    .into_array();
    Ok(outer(structure(child, Validity::NonNullable)?))
}

#[test]
fn unused_lazy_varbin_offsets_and_referenced_errors() -> VortexResult<()> {
    for different in [false, true] {
        let build = || lazy_varbin_offsets(7, different);
        let mut ctx = array_session().create_execution_ctx();
        assert_eq!(
            nested_list_elements_are_constant(&build()?, 2, &mut ctx)?,
            None
        );
        compare("unused lazy binary offsets", build, Some(!different))?;
    }
    // Index 3 is read by the first pair; index 5 is read only by the remaining pair.
    for error_index in [3, 5] {
        let build = || lazy_varbin_offsets(error_index, false);
        let mut ctx = array_session().create_execution_ctx();
        assert!(original_list(&build()?, &mut ctx).is_err());
        compare("referenced lazy binary offset error", build, None)?;
    }
    Ok(())
}

#[test]
fn lazy_unused_values_after_equal_first_pair_remain_unobserved() -> VortexResult<()> {
    compare(
        "equal first pair before unused lazy value",
        || {
            let values = PrimitiveArray::from_iter([1i32, 2, 1, 2, 1, 2, 9]).into_array();
            let divisors = PrimitiveArray::from_iter([1i32, 1, 1, 1, 1, 1, 0]).into_array();
            Ok(outer(structure(
                values.binary(divisors, Operator::Div)?,
                Validity::NonNullable,
            )?))
        },
        Some(true),
    )
}

#[test]
fn zy283_all_outer_offset_widths_and_mismatch_positions() -> VortexResult<()> {
    macro_rules! width {
        ($ty:ty) => {
            for mismatch in [None, Some(1usize), Some(2), Some(3), Some(5)] {
                for offsets in [
                    [2usize, 4, 6, 8, 10, 12],
                    [2, 2, 6, 4, 8, 10],
                    [2, 3, 4, 5, 6, 7],
                ] {
                    compare(
                        stringify!($ty),
                        || {
                            let mut values = [1i32; 16];
                            if let Some(index) = mismatch {
                                values[offsets[index]] = 2;
                            }
                            let offsets = PrimitiveArray::from_iter(offsets.map(|i| {
                                let offset: $ty = i.as_();
                                offset
                            }))
                            .into_array();
                            ListViewArray::try_new(
                                structure(
                                    PrimitiveArray::from_iter(values).into_array(),
                                    Validity::NonNullable,
                                )?,
                                offsets,
                                PrimitiveArray::from_iter([1u64; 6]).into_array(),
                                Validity::NonNullable,
                            )
                        },
                        None,
                    )?;
                }
            }
        };
    }
    width!(u8);
    width!(u16);
    width!(u32);
    width!(u64);
    width!(i8);
    width!(i16);
    width!(i32);
    width!(i64);
    Ok(())
}
