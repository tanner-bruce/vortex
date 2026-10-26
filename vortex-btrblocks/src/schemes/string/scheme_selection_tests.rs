// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests to verify that each string compression scheme produces the expected encoding.

use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_fsst::FSST;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::BtrBlocksCompressor;
use crate::BtrBlocksCompressorBuilder;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

#[test]
fn test_constant_compressed() -> VortexResult<()> {
    let strings: Vec<Option<&str>> = vec![Some("constant_value"); 100];
    let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
    let array_ref = array.into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Constant>());
    Ok(())
}

#[test]
fn test_dict_compressed() -> VortexResult<()> {
    let distinct_values = ["apple", "banana", "cherry"];
    let mut strings = Vec::with_capacity(1000);
    for i in 0..1000 {
        strings.push(Some(distinct_values[i % 3]));
    }
    let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
    let array_ref = array.into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert!(compressed.is::<Dict>());
    Ok(())
}

#[test]
fn test_all_schemes_includes_onpair() {
    use crate::SchemeExt;
    use crate::schemes::string::onpair::OnPairScheme;

    let ids: Vec<_> = crate::ALL_SCHEMES.iter().map(|s| s.id()).collect();
    assert!(
        ids.contains(&OnPairScheme.id()),
        "OnPairScheme not registered in ALL_SCHEMES"
    );
}

#[test]
fn test_default_btrblocks_compressor_selects_onpair() -> VortexResult<()> {
    // Dictionary-style string corpus: high lexical overlap, short rows.
    // OnPair beats FSST on this corpus, so it wins the sample-based
    // comparison even though both are registered.
    let mut strings = Vec::with_capacity(1000);
    for i in 0..1000 {
        strings.push(Some(format!(
            "this_is_a_common_prefix_with_some_variation_{i}_and_a_common_suffix_pattern"
        )));
    }
    let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
    let array_ref = array.into_array();
    let compressed =
        BtrBlocksCompressor::default().compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert!(
        compressed.is::<vortex_onpair::OnPair>(),
        "expected OnPair, got {}",
        compressed.encoding_id()
    );
    Ok(())
}

/// FSST is registered in the default scheme list, and an FSST-only builder
/// still produces an FSST array.
#[test]
fn test_fsst_in_default_scheme_list() -> VortexResult<()> {
    use crate::BtrBlocksCompressorBuilder;
    use crate::SchemeExt;
    use crate::schemes::string::FSSTScheme;

    // FSST is registered by default.
    assert!(
        crate::ALL_SCHEMES.iter().any(|s| s.id() == FSSTScheme.id()),
        "FSSTScheme should be in ALL_SCHEMES",
    );

    // An FSST-only builder still produces an FSST array for FSST-favourable
    // input.
    let mut strings = Vec::with_capacity(1000);
    for i in 0..1000 {
        strings.push(Some(format!(
            "this_is_a_common_prefix_with_some_variation_{i}_and_a_common_suffix_pattern"
        )));
    }
    let array = VarBinViewArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));
    let array_ref = array.into_array();

    let compressor = BtrBlocksCompressorBuilder::empty()
        .with_new_scheme(&FSSTScheme)
        .build();
    let compressed = compressor.compress(&array_ref, &mut SESSION.create_execution_ctx())?;
    assert!(
        compressed.is::<FSST>(),
        "expected FSST when only FSSTScheme is registered, got {}",
        compressed.encoding_id()
    );
    Ok(())
}

#[test]
fn dictionary_reuse_uses_consistent_samples_and_prunes_unused_values() -> VortexResult<()> {
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
            let flat_arrow = arrow_array::StringArray::from_iter(
                (0..8192).map(|i| (i % stride == 0).then(|| values[(i / stride) % 16].as_str())),
            );
            let codes = arrow_array::UInt32Array::from_iter((0..8192).map(|i| {
                (i % stride == 0)
                    .then_some(u32::try_from((i / stride) % 16).expect("16 dictionary values"))
            }));
            let dict_arrow =
                arrow_array::DictionaryArray::<arrow_array::types::UInt32Type>::try_new(
                    codes,
                    Arc::new(arrow_array::StringArray::from_iter_values(values)),
                )?;
            let flat = SESSION
                .arrow()
                .from_arrow_array(Arc::new(flat_arrow), true)?;
            let dict = SESSION
                .arrow()
                .from_arrow_array(Arc::new(dict_arrow), true)?;
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

#[test]
fn dictionary_reuse_recurses_into_struct_lists_and_respects_allowed_encodings() -> VortexResult<()>
{
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
    )?
    .into_array();
    let allowed = [Dict.id()].into_iter().collect();
    let compressor = BtrBlocksCompressorBuilder::default()
        .retain_allowed_encodings(&allowed)
        .with_string_dictionary_reuse(true)
        .build();
    let output = compressor.compress(&input, &mut ctx)?;
    assert_arrays_eq!(&input, &output, &mut ctx);
    let list = output.execute::<ListViewArray>(&mut ctx)?;
    let entries = list.elements().clone().execute::<StructArray>(&mut ctx)?;
    assert!(entries.unmasked_field(0).is::<Dict>());
    let fallback = BtrBlocksCompressorBuilder::default()
        .retain_allowed_encodings(&HashSet::new())
        .with_string_dictionary_reuse(true)
        .build()
        .compress(&input, &mut ctx)?;
    assert_arrays_eq!(&input, &fallback, &mut ctx);
    let list = fallback.execute::<ListViewArray>(&mut ctx)?;
    let entries = list.elements().clone().execute::<StructArray>(&mut ctx)?;
    assert!(!entries.unmasked_field(0).is::<Dict>());
    Ok(())
}
