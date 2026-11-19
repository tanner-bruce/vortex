// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare the real Sparse scheme's serialized output across fork revisions.

use std::fmt::Write as _;

use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::serde::SerializeOptions;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt as _;

use super::SparseScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::GenerateStatsOptions;
use crate::Scheme;

/// Admits a tie so the test can inspect the fill chosen by the real Sparse encoder.
#[derive(Debug)]
struct SparseOnTie;

impl Scheme for SparseOnTie {
    fn scheme_name(&self) -> &'static str {
        SparseScheme.scheme_name()
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        SparseScheme.matches(canonical)
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        SparseScheme.produced_encodings()
    }

    fn stats_options(&self) -> GenerateStatsOptions {
        SparseScheme.stats_options()
    }

    fn num_children(&self) -> usize {
        SparseScheme.num_children()
    }

    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        SparseScheme.descendant_exclusions()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _ctx: CompressorContext,
        _exec: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(EstimateVerdict::AlwaysUse)
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        ctx: CompressorContext,
        exec: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        SparseScheme.compress(compressor, data, ctx, exec)
    }
}

fn report_sparse(
    name: &str,
    array: ArrayRef,
    fill: Scalar,
    scheme: &'static dyn Scheme,
) -> VortexResult<()> {
    let session = VortexSession::empty();
    vortex_sparse::initialize(&session);
    let mut ctx = session.create_execution_ctx();
    let compressor = CascadingCompressor::new(vec![scheme]);
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
            report_sparse(concat!(stringify!($ty), "-tied"), tied, null_fill, &SparseScheme)?;
            let dominant = PrimitiveArray::from_iter((0..256).map(|index| {
                if index % 23 == 0 { two } else { one }
            })).into_array();
            report_sparse(
                concat!(stringify!($ty), "-dominant"), dominant,
                Scalar::primitive(one, Nullability::NonNullable), &SparseScheme,
            )?;
        )+};
    }
    check!(u8, u16, u32, u64, i8, i16, i32, i64);

    let tie = PrimitiveArray::from_iter((0..256u64).map(|i| 1 + i % 2));
    report_sparse(
        "u64-forced-tied",
        tie.into_array(),
        Scalar::primitive(1u64, Nullability::NonNullable),
        &SparseOnTie,
    )
}
