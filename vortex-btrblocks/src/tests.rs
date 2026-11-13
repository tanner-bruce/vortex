// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression round trips and regressions involving the built-in schemes.
//!
//! These tests cover scheme composition and configuration. Individual encoding kernels and the
//! generic selection algorithm are tested in their owning crates.

use std::iter;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rstest::rstest;
#[cfg(feature = "zstd")]
use vortex_array::ArrayId;
#[cfg(feature = "zstd")]
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::Nullability;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::validity::Validity;
use vortex_datetime_parts::DateTimeParts;
#[cfg(feature = "zstd")]
use vortex_edition::EDITION_DECLARATIONS;
#[cfg(feature = "zstd")]
use vortex_edition::EDITION_FAMILIES;
#[cfg(feature = "zstd")]
use vortex_edition::EditionId;
#[cfg(feature = "zstd")]
use vortex_edition::EditionSessionExt;
#[cfg(feature = "zstd")]
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexResult;
use vortex_fastlanes::Delta;
#[cfg(feature = "zstd")]
use vortex_session::VortexSession;

use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorBuilder;
use crate::SESSION;
use crate::schemes::string::StringDictScheme;

#[track_caller]
fn assert_roundtrip(compressor: &BtrBlocksCompressor, input: &ArrayRef) -> VortexResult<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressed = compressor.compress(input, &mut ctx)?;
    assert_arrays_eq!(&compressed, input, &mut ctx);

    Ok(compressed)
}

#[test]
fn test_default_compressor_roundtrip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let integers = PrimitiveArray::from_option_iter(
        (0..2048i32).map(|i| (i % 7 != 0).then_some(1_000_000 + (i * 37) % 100)),
    );
    let floats = PrimitiveArray::from_iter((0..2048).map(|i| f64::from(i) / 100.0));
    let strings = VarBinViewArray::from_iter(
        (0..2048).map(|i| (i % 7 != 0).then(|| format!("https://example.com/products/{i}"))),
        DType::Utf8(Nullability::Nullable),
    );
    let binary = VarBinViewArray::from_iter(
        (0..2048).map(|i| (i % 7 != 0).then(|| format!("PREFIX_{i:09}").into_bytes())),
        DType::Binary(Nullability::Nullable),
    );
    let decimals = DecimalArray::new(
        (0..2048i64).map(|i| i * 123).collect(),
        DecimalDType::new(12, 2),
        Validity::NonNullable,
    );
    let timestamps = TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..2048i64).map(|i| 1_700_000_000_000_000 + i * 123_456))
            .into_array(),
        TimeUnit::Microseconds,
        None,
    );
    let elements = StructArray::from_fields(&[
        ("integer", integers.into_array()),     //
        ("float", floats.into_array()),         //
        ("string", strings.into_array()),       //
        ("binary", binary.into_array()),        //
        ("decimal", decimals.into_array()),     //
        ("timestamp", timestamps.into_array()), //
    ])?;
    let offsets = PrimitiveArray::from_iter((0..=2048i32).step_by(4));
    let input = ListArray::try_new(
        elements.into_array(),
        offsets.into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();

    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.nbytes() < input.nbytes());

    Ok(())
}

#[rstest]
#[case::seconds(TimeUnit::Seconds)]
#[case::milliseconds(TimeUnit::Milliseconds)]
fn test_extreme_timestamps_roundtrip(#[case] unit: TimeUnit) -> VortexResult<()> {
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let input = TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..2048i64).map(|i| match i % 3 {
            0 => i64::MAX - i,
            1 => i64::MIN + i,
            _ => 1_700_000_000 + i,
        }))
        .into_array(),
        unit,
        None,
    )
    .into_array();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.is::<DateTimeParts>());

    Ok(())
}

#[rstest]
#[case::unaligned({
    let mut rng = StdRng::seed_from_u64(7);
    let mut value = 500_000i32;
    PrimitiveArray::from_iter((0..1025).map(|_| {
        value += 1 + (rng.next_u32() % 6) as i32;
        value
    }))
})]
#[case::nullable_unaligned(
    PrimitiveArray::from_option_iter(iter::once(None).chain((1i32..=100_000).map(Some)))
)]
fn test_delta_unaligned_roundtrip(#[case] input: PrimitiveArray) -> VortexResult<()> {
    // Zero-padding the final chunk used to inflate the delta span and reject this scheme.
    let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
        .unrestricted()
        .build();
    let input = input.into_array();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert!(compressed.is::<Delta>());

    let mut ctx = SESSION.create_execution_ctx();
    assert_eq!(sum(&compressed, &mut ctx)?, sum(&input, &mut ctx)?);

    Ok(())
}

#[cfg(feature = "zstd")]
#[rstest]
#[case::array_level(CORE_2026_08_3, vortex_zstd::Zstd.id())]
#[case::buffer_level(vortex_zstd::editions::ZSTD_2026_02, vortex_zstd::ZstdBuffers.id())]
fn test_cuda_binary_zstd_follows_editions(
    #[case] edition: EditionId,
    #[case] expected: ArrayId,
) -> VortexResult<()> {
    let values: Vec<_> = (0..1024u32)
        .map(|i| {
            let mut value = Vec::from(&b"common binary payload prefix "[..]);
            value.extend_from_slice(&i.to_le_bytes());
            value.extend_from_slice(&[b'x'; 96]);
            value
        })
        .collect();
    let input = VarBinViewArray::from_iter_bin(values.iter().map(Vec::as_slice)).into_array();

    // The CUDA preset enables both Zstd schemes; the edition filter must choose between them.
    let session = VortexSession::empty();
    vortex_zstd::initialize(&session);
    for family in EDITION_FAMILIES {
        session.editions().declare_family(family)?;
    }
    for declaration in EDITION_DECLARATIONS {
        session.register_edition(declaration)?;
    }
    session.enable_edition(edition)?;
    let compressor = BtrBlocksCompressorBuilder::from_session(&session)
        .only_cuda_compatible()
        .build();
    let compressed = assert_roundtrip(&compressor, &input)?;
    assert_eq!(compressed.encoding_id(), expected);

    Ok(())
}

/// Reusing an existing Utf8 dictionary must select the same encoding as compressing the
/// equivalent flat strings, across sample strides and value lengths.
#[test]
fn dictionary_reuse_matches_flat_selection_across_sample_strides() -> VortexResult<()> {
    use vortex_array::arrays::DictArray;

    let mut ctx = SESSION.create_execution_ctx();
    let compressor = BtrBlocksCompressorBuilder::default()
        .with_string_dictionary_reuse(true)
        .build();
    for stride in [1, 20, 100, 200, 1000] {
        for length in [8, 48, 1024] {
            let values: Vec<String> = (1..=16)
                .map(|seed| {
                    let mut state = seed as u64;
                    (0..length)
                        .map(|_| {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            char::from(b'!' + (state % 90) as u8)
                        })
                        .collect()
                })
                .collect();
            let flat = VarBinViewArray::from_iter(
                (0..8192).map(|i| (i % stride == 0).then(|| values[(i / stride) % 16].as_str())),
                DType::Utf8(Nullability::Nullable),
            )
            .into_array();
            let codes = PrimitiveArray::from_option_iter((0..8192).map(|i| {
                (i % stride == 0)
                    .then_some(u32::try_from((i / stride) % 16).expect("16 dictionary values"))
            }))
            .into_array();
            let dict =
                DictArray::try_new(codes, VarBinViewArray::from_iter_str(values).into_array())?
                    .into_array();
            let a = compressor.compress(&flat, &mut ctx)?;
            let b = compressor.compress(&dict, &mut ctx)?;
            assert_arrays_eq!(&flat, &b, &mut ctx);
            assert_arrays_eq!(&a, &b, &mut ctx);
            assert_eq!(
                a.encoding_id(),
                b.encoding_id(),
                "stride={stride} length={length}"
            );
        }
    }
    Ok(())
}

/// Reuse recurses into struct and list children, preserving their dictionaries.
#[test]
fn dictionary_reuse_recurses_into_struct_lists() -> VortexResult<()> {
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::ListViewArray;

    let mut ctx = SESSION.create_execution_ctx();
    let dictionary = DictArray::try_new(
        PrimitiveArray::from_iter((0..4096).map(|i| i % 2_u32)).into_array(),
        VarBinViewArray::from_iter_str([
            "first repeated long value",
            "second repeated long value",
            "unused",
        ])
        .into_array(),
    )?
    .into_array();
    let input = ListArray::try_new(
        StructArray::try_new(
            ["value"].into(),
            vec![dictionary],
            4096,
            Validity::NonNullable,
        )?
        .into_array(),
        PrimitiveArray::from_iter([0_u32, 2048, 4096]).into_array(),
        Validity::NonNullable,
        &mut ctx,
    )?
    .into_array();
    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&StringDictScheme)
        .with_string_dictionary_reuse(true)
        .build();
    let output = compressor.compress(&input, &mut ctx)?;
    assert_arrays_eq!(&input, &output, &mut ctx);
    let list = output.execute::<ListViewArray>(&mut ctx)?;
    let entries = list.elements().clone().execute::<StructArray>(&mut ctx)?;
    assert!(entries.unmasked_field(0).is::<Dict>());

    let fallback = BtrBlocksCompressorBuilder::empty()
        .with_string_dictionary_reuse(true)
        .build()
        .compress(&input, &mut ctx)?;
    assert_arrays_eq!(&input, &fallback, &mut ctx);
    let list = fallback.execute::<ListViewArray>(&mut ctx)?;
    let entries = list.elements().clone().execute::<StructArray>(&mut ctx)?;
    assert!(!entries.unmasked_field(0).is::<Dict>());
    Ok(())
}
