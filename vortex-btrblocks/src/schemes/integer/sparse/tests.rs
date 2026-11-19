// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare the real Sparse scheme's serialized output across fork revisions.

use std::fmt::Write as _;

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::serde::SerializeOptions;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;

use super::SparseScheme;
use crate::CascadingCompressor;

fn report_sparse(name: &str, array: ArrayRef) -> VortexResult<()> {
    let session = VortexSession::empty();
    vortex_sparse::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let compressor = CascadingCompressor::new(vec![&SparseScheme]);
    let compressed = compressor.compress(&array, &mut ctx)?;
    assert!(compressed.is::<Sparse>());
    assert_arrays_eq!(&array, &compressed, &mut ctx);

    let buffers = compressed.serialize(
        &ArrayContext::empty(),
        &session,
        &SerializeOptions::default(),
    )?;
    let mut bytes = String::new();
    for byte in buffers.iter().flat_map(|buffer| buffer.as_slice()) {
        write!(bytes, "{byte:02x}").vortex_expect("writing to a String cannot fail");
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
            report_sparse(concat!(stringify!($ty), "-tied"), tied)?;
            let dominant = PrimitiveArray::from_iter((0..256).map(|index| {
                if index % 23 == 0 { two } else { one }
            })).into_array();
            report_sparse(concat!(stringify!($ty), "-dominant"), dominant)?;
        )+};
    }
    check!(u8, u16, u32, u64, i8, i16, i32, i64);
    Ok(())
}
