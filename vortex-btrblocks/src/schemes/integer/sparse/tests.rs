// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare the real Sparse scheme's serialized output across fork revisions.

use std::fmt::Write as _;
use std::sync::LazyLock;

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::serde::SerializeOptions;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;

use super::SparseScheme;
use crate::CascadingCompressor;
use crate::PinSpec;
use crate::SchemePins;

fn report_sparse(
    name: &str,
    array: ArrayRef,
    fill: Scalar,
    pins: Option<&'static SchemePins>,
) -> VortexResult<()> {
    let session = VortexSession::empty();
    vortex_sparse::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let mut compressor = CascadingCompressor::new(vec![&SparseScheme]);
    if let Some(pins) = pins {
        compressor = compressor.with_pins(pins);
    }
    let compressed = compressor.compress(&array, &mut ctx)?;
    let sparse = compressed
        .as_opt::<Sparse>()
        .vortex_expect("Sparse should win");
    assert_eq!(sparse.fill_scalar(), &fill);
    assert_arrays_eq!(&array, &compressed, &mut ctx);

    let buffers = compressed.serialize(
        &ArrayContext::empty(),
        &session,
        &SerializeOptions::default(),
    )?;
    let mut bytes = String::new();
    for byte in buffers.iter().flat_map(|buffer| buffer.as_slice()) {
        assert!(write!(bytes, "{byte:02x}").is_ok());
    }
    eprintln!("VORTEX_SPARSE_PARITY {name} {bytes}");
    Ok(())
}

#[test]
fn sparse_scheme_preserves_baseline_bytes() -> VortexResult<()> {
    macro_rules! check {
        ($($ty:ty),+) => {$(
            let one: $ty = 1;
            let two: $ty = 2;
            // Both values are equally frequent. The null fraction keeps Sparse eligible.
            let tied = PrimitiveArray::from_option_iter((0..256).map(|index| match index {
                12 => Some(one),
                137 => Some(two),
                _ => None,
            })).into_array();
            let null_fill = Scalar::null(tied.dtype().clone());
            report_sparse(concat!(stringify!($ty), "-tied"), tied, null_fill, None)?;
            let dominant = PrimitiveArray::from_iter((0..256).map(|index| {
                if index % 23 == 0 { two } else { one }
            })).into_array();
            report_sparse(
                concat!(stringify!($ty), "-dominant"), dominant,
                Scalar::primitive(one, Nullability::NonNullable), None,
            )?;
        )+};
    }
    check!(u8, u16, u32, u64, i8, i16, i32, i64);

    // Pin Sparse so it encodes a two-way tie without the null-dominated fill overriding the winner.
    static PINS: LazyLock<SchemePins> = LazyLock::new(|| {
        SchemePins::new(
            &CascadingCompressor::new(vec![&SparseScheme]),
            &[PinSpec {
                path: &[],
                history: &[],
                decision: Some("vortex.int.sparse"),
            }],
        )
        .vortex_expect("the root Sparse pin resolves")
    });
    let pinned_tie = PrimitiveArray::from_iter((0..256u64).map(|i| 1 + i % 2));
    report_sparse(
        "u64-pinned-tied",
        pinned_tie.into_array(),
        Scalar::primitive(1u64, Nullability::NonNullable),
        Some(&PINS),
    )
}
