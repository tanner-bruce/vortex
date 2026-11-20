// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Check exact size and encoded bytes across the cascade's canonicalization.

use std::fmt::Write as _;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::CanonicalValidity;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::extension::datetime::Date;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::serde::SerializeOptions;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
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
        .into_array().slice(1..3).vortex_expect("slice is within the fixture")
)]
#[case::views(
    VarBinViewArray::from_iter_nullable_str([Some("inline"), None, Some(OUT_OF_LINE)])
        .into_array()
)]
#[case::views_all_null(VarBinViewArray::from_iter_nullable_str([None::<&str>; 4]).into_array())]
#[case::views_slice({
    let unused = "x".repeat(8192);
    VarBinViewArray::from_iter_str([unused.as_str(), OUT_OF_LINE, "inline", unused.as_str()])
        .into_array().slice(1..3).vortex_expect("slice is within the fixture")
})]
#[case::dictionary(
    DictArray::try_new(
        PrimitiveArray::from_iter([0u32, 2, 0, 2, 2]).into_array(),
        VarBinViewArray::from_iter_str([OUT_OF_LINE, "unreferenced", "short"]).into_array(),
    ).vortex_expect("the dictionary codes reference existing strings").into_array()
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

fn chunked<T: NativePType>(rows: impl IntoIterator<Item = Option<T>>) -> VortexResult<ArrayRef> {
    ChunkedArray::try_new(
        rows.into_iter()
            .map(|row| PrimitiveArray::from_option_iter([row]).into_array()),
        DType::Primitive(T::PTYPE, Nullability::Nullable),
    )
    .map(IntoArray::into_array)
}

/// The extension case compares storage size and wire bytes. The unrestricted carry also
/// passes this case because compression measures the extension storage again.
#[rstest]
#[case::chunked("chunked", 32, 18)]
#[case::struct_field("struct", 32, 18)]
#[case::extension_storage("extension", 80, 66)]
#[case::dict_values("dict", 32, 18)]
fn chunked_size_and_compression_match_baseline(
    #[case] case: &str,
    #[case] split_size: usize,
    #[case] fresh_size: usize,
) -> VortexResult<()> {
    let rows = || (0..16u8).map(|i| (i % 2 == 0).then_some(i % 4));
    let (input, flat) = match case {
        "chunked" => (
            chunked(rows())?,
            PrimitiveArray::from_option_iter(rows()).into_array(),
        ),
        "struct" => (
            StructArray::from_fields(&[("field", chunked(rows())?)])?.into_array(),
            StructArray::from_fields(&[(
                "field",
                PrimitiveArray::from_option_iter(rows()).into_array(),
            )])?
            .into_array(),
        ),
        "dict" => (
            DictArray::try_new(
                PrimitiveArray::from_iter(0..16u32).into_array(),
                chunked(rows())?,
            )?
            .into_array(),
            PrimitiveArray::from_option_iter(rows()).into_array(),
        ),
        "extension" => {
            let rows = || rows().map(|row| row.map(i32::from));
            let dtype = Date::new(TimeUnit::Days, Nullability::Nullable).erased();
            (
                ExtensionArray::new(dtype.clone(), chunked(rows())?).into_array(),
                ExtensionArray::new(dtype, PrimitiveArray::from_option_iter(rows()).into_array())
                    .into_array(),
            )
        }
        _ => unreachable!("unknown compression fixture"),
    };

    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(uncompressed_size_in_bytes(&input, &mut ctx)?, split_size);
    let compressor = CascadingCompressor::new(vec![&IntDictScheme, &StringDictScheme]);
    let encoded = compressor.compress(&input, &mut ctx)?;
    let flat_encoded = compressor.compress(&flat, &mut ctx)?;
    let encoding_ids = |array: &ArrayRef| {
        array
            .depth_first_traversal()
            .map(|child| child.encoding_id().to_string())
            .collect::<Vec<_>>()
    };
    let encodings = encoding_ids(&encoded);
    assert_eq!(encodings, encoding_ids(&flat_encoded));

    let context = ArrayContext::empty();
    let options = SerializeOptions::default();
    let bytes = encoded.serialize(&context, &SESSION, &options)?;
    assert_eq!(bytes, flat_encoded.serialize(&context, &SESSION, &options)?);

    let canonical = flat
        .execute::<CanonicalValidity>(&mut ctx)?
        .0
        .compact(&mut ctx)?
        .into_array();
    canonical.statistics().clear(Stat::UncompressedSizeInBytes);
    let size = uncompressed_size_in_bytes(&canonical, &mut ctx)?;
    assert_eq!(size, fresh_size);

    // The baseline checkout runs this same fixture and compares the full wire bytes.
    let mut wire = String::new();
    for byte in bytes.iter().flat_map(|buffer| buffer.as_slice()) {
        assert!(write!(wire, "{byte:02x}").is_ok());
    }
    let encodings = encodings.join(",");
    eprintln!("VORTEX_CASCADE_PARITY {case} {split_size} {size} {encodings} {wire}");
    Ok(())
}
