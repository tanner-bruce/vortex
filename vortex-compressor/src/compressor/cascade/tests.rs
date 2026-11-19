// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Check exact size and encoded bytes across the cascade's canonicalization.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::CanonicalValidity;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::serde::SerializeOptions;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::CascadingCompressor;
use super::CompressorContext;
use crate::builtins::IntDictScheme;
use crate::builtins::StringDictScheme;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

const UTF8: DType = DType::Utf8(Nullability::Nullable);
const OUT_OF_LINE: &str = "an outlined string for the exact size test";

#[rstest]
#[case::primitive(
    PrimitiveArray::from_option_iter((0..128).map(|i| (i % 5 != 0).then_some(i % 2)))
        .into_array()
)]
#[case::bool(BoolArray::from_iter([true, false, true, true, false]).into_array())]
#[case::decimal(
    DecimalArray::new(
        buffer![100i128, 5, 200, 300],
        DecimalDType::new(9, 2),
        Validity::from_iter([true, false, true, true]),
    ).into_array()
)]
#[case::varbin(
    VarBinArray::from_iter([Some("inline"), None, Some(OUT_OF_LINE)], UTF8).into_array()
)]
#[case::varbin_all_null(
    VarBinArray::from_iter([None::<&str>, None], UTF8).into_array()
)]
#[case::varbin_slice(
    VarBinArray::from_iter([Some("prefix"), None, Some(OUT_OF_LINE), Some("suffix")], UTF8)
        .into_array().slice(1..3).unwrap()
)]
#[case::views(
    VarBinViewArray::from_iter_nullable_str([Some("inline"), None, Some(OUT_OF_LINE)])
        .into_array()
)]
#[case::views_all_null(VarBinViewArray::from_iter_nullable_str([None::<&str>; 4]).into_array())]
#[case::views_slice({
    let unused = "x".repeat(8192);
    VarBinViewArray::from_iter_str([unused.as_str(), OUT_OF_LINE, "inline", unused.as_str()])
        .into_array().slice(1..3).unwrap()
})]
#[case::dictionary(
    DictArray::try_new(
        PrimitiveArray::from_iter([0u32, 2, 0, 2, 2]).into_array(),
        VarBinViewArray::from_iter_str([OUT_OF_LINE, "unreferenced", "short"]).into_array(),
    ).unwrap().into_array()
)]
#[case::constant(ConstantArray::new(OUT_OF_LINE.to_owned(), 43).into_array())]
fn exact_size_survives_canonicalization(#[case] input: ArrayRef) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let size = uncompressed_size_in_bytes(&input, &mut ctx)?;
    let canonical = input
        .clone()
        .execute::<CanonicalValidity>(&mut ctx)?
        .0
        .compact(&mut ctx)?
        .into_array();
    canonical.statistics().clear(Stat::UncompressedSizeInBytes);
    assert_eq!(size, uncompressed_size_in_bytes(&canonical, &mut ctx)?);

    // Enter below compress_at, whose initial size lookup would populate the baseline's cache.
    // With no input size, selection must measure the normalized array as it did before.
    input.statistics().clear(Stat::UncompressedSizeInBytes);
    let compressor = CascadingCompressor::new(vec![&IntDictScheme, &StringDictScheme]);
    let measured = compressor.compress_input(&input, CompressorContext::new(), &mut ctx)?;
    input.statistics().set(
        Stat::UncompressedSizeInBytes,
        Precision::Exact((size as u64).into()),
    );
    let carried = compressor.compress_input(&input, CompressorContext::new(), &mut ctx)?;

    let context = ArrayContext::empty();
    let options = SerializeOptions::default();
    assert_eq!(
        measured.serialize(&context, &SESSION, &options)?,
        carried.serialize(&context, &SESSION, &options)?,
    );
    Ok(())
}
