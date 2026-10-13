// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::super::ordering_predicate;
use super::*;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::bool::BoolArrayExt;
use crate::dtype::FieldNames;
use crate::dtype::MapDType;
use crate::dtype::PType;
use crate::validity::Validity;

fn list(
    values: ArrayRef,
    offsets: impl IntoIterator<Item = u64>,
    sizes: impl IntoIterator<Item = u64>,
) -> ArrayRef {
    ListViewArray::new(
        values,
        PrimitiveArray::from_iter(offsets).into_array(),
        PrimitiveArray::from_iter(sizes).into_array(),
        Validity::NonNullable,
    )
    .into_array()
}
fn tree(kind: &str, rows: usize) -> VortexResult<ArrayRef> {
    let child = list(
        PrimitiveArray::from_iter(
            (0..rows * 2).map(|i| [1.0, -0.0, f64::from_bits(0x7ff8_0000_0000_0001), 3.0][i % 4]),
        )
        .into_array(),
        (0..rows).map(|i| (i * 2) as u64),
        (0..rows).map(|_| 2),
    );
    Ok(match kind {
        "struct" => StructArray::try_new(
            FieldNames::from(["x"]),
            vec![child],
            rows,
            Validity::NonNullable,
        )?
        .into_array(),
        "map" => {
            let entries = StructArray::try_new(
                FieldNames::from(["key", "value"]),
                vec![
                    PrimitiveArray::from_iter((0..rows).map(|_| 1i32)).into_array(),
                    child,
                ],
                rows,
                Validity::NonNullable,
            )?
            .into_array();
            MapArray::try_new(
                MapDType::try_new(
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::List(
                        std::sync::Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
                        Nullability::NonNullable,
                    ),
                    false,
                )?,
                ListViewArray::new(
                    entries,
                    PrimitiveArray::from_iter((0..rows).map(|i| i as u64)).into_array(),
                    PrimitiveArray::from_iter((0..rows).map(|_| 1u64)).into_array(),
                    Validity::NonNullable,
                ),
            )?
            .into_array()
        }
        "fsl" => FixedSizeListArray::try_new(
            PrimitiveArray::from_iter((0..rows * 2).map(|i| {
                let value: i32 = (i % 4).as_();
                value
            }))
            .into_array(),
            2,
            Validity::NonNullable,
            rows,
        )?
        .into_array(),
        _ => child,
    })
}
#[test]
fn zy283_identity_comparator_differential() -> VortexResult<()> {
    for kind in ["list", "struct", "map", "fsl"] {
        let lhs = tree(kind, 7)?;
        let rhs = tree(kind, 7)?;
        assert!(!ArrayRef::ptr_eq(&lhs, &rhs));
        let mut ctx = array_session().create_execution_ctx();
        let same = build_comparator(&lhs, &lhs, &mut ctx)?;
        let distinct = build_comparator(&lhs, &rhs, &mut ctx)?;
        for i in 0..7 {
            for j in 0..7 {
                assert_eq!(same(i, j), distinct(i, j), "{kind} {i}/{j}");
            }
        }
        // Shared buffers and separate slice indexing must remain distinct.
        let left = lhs
            .slice(0..6)?
            .execute::<RecursiveCanonical>(&mut ctx)?
            .0
            .into_array();
        let right = lhs
            .slice(1..7)?
            .execute::<RecursiveCanonical>(&mut ctx)?
            .0
            .into_array();
        let slices = build_comparator(&left, &right, &mut ctx)?;
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(slices(i, j), distinct(i, j + 1));
            }
        }
        for op in [
            CompareOperator::Eq,
            CompareOperator::NotEq,
            CompareOperator::Gt,
            CompareOperator::Gte,
            CompareOperator::Lt,
            CompareOperator::Lte,
        ] {
            let result = compare_nested(&left, &right, op, Nullability::NonNullable, &mut ctx)?
                .execute::<BoolArray>(&mut ctx)?;
            let predicate = ordering_predicate(op);
            for i in 0..6 {
                assert_eq!(
                    result.to_bit_buffer().value(i),
                    predicate(slices(i, i)),
                    "{kind} {op:?} {i}"
                );
            }
        }
    }
    // Varying lengths and nested nulls: self identity is not row equality.
    let lhs = ListViewArray::new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(1), Some(2), None]).into_array(),
        PrimitiveArray::from_iter([0u64, 0, 2, 4]).into_array(),
        PrimitiveArray::from_iter([0u64, 2, 2, 1]).into_array(),
        Validity::from_iter([true, true, true, false]),
    )
    .into_array();
    let typed = lhs
        .clone()
        .execute::<ListViewArray>(&mut array_session().create_execution_ctx())?;
    let rhs = ListViewArray::new(
        typed.elements().clone(),
        typed.offsets().clone(),
        typed.sizes().clone(),
        typed.validity()?,
    )
    .into_array();
    assert!(!ArrayRef::ptr_eq(&lhs, &rhs));
    let mut ctx = array_session().create_execution_ctx();
    let same = build_comparator(&lhs, &lhs, &mut ctx)?;
    let distinct = build_comparator(&lhs, &rhs, &mut ctx)?;
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(same(i, j), distinct(i, j));
        }
    }
    assert_ne!(same(0, 1), Ordering::Equal);
    Ok(())
}
