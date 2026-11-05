// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheme pins with the BtrBlocks schemes: replayed schemes decode to their input, and the sites
//! the search observes are the sites the pin lookup keys.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::TemporalArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::extension::datetime::TimeUnit;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BtrBlocksCompressorBuilder;
use crate::PinSpec;
use crate::PinTally;
use crate::SchemePins;
use crate::SiteKey;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

const ROWS: i64 = 4096;

/// The default schemes, with Zstd for strings when it is built, as zany compacts.
fn builder() -> BtrBlocksCompressorBuilder {
    let builder = BtrBlocksCompressorBuilder::default();
    #[cfg(feature = "zstd")]
    let builder = builder.with_compact();
    builder
}

fn sequence() -> ArrayRef {
    PrimitiveArray::from_iter(0..ROWS).into_array()
}

fn runs() -> ArrayRef {
    PrimitiveArray::from_iter((0..ROWS).map(|i| (i / 64) as i32 * 7)).into_array()
}

fn offset_ints() -> ArrayRef {
    PrimitiveArray::from_iter((0..ROWS).map(|i| 1_000_000_000_000 + (i * 7_919) % 100)).into_array()
}

fn small_ints() -> ArrayRef {
    PrimitiveArray::from_iter((0..ROWS).map(|i| ((i * 7_919) % 1_000) as u32)).into_array()
}

fn few_values() -> ArrayRef {
    let values = [1_000_003_i64, 7, 123_456_789, 42, 99];
    PrimitiveArray::from_iter((0..ROWS).map(|i| values[(i * 31 % 5) as usize])).into_array()
}

fn timestamps() -> ArrayRef {
    TemporalArray::new_timestamp(
        PrimitiveArray::from_iter((0..ROWS).map(|i| 1_793_000_000_000_000 + i * 1_000_003))
            .into_array(),
        TimeUnit::Microseconds,
        None,
    )
    .into_array()
}

fn strings() -> ArrayRef {
    VarBinViewArray::from_iter_str((0..ROWS).map(|i| format!("service-{}-span", i * 7_919 % 37)))
        .into_array()
}

/// An attribute-like column: lists of `(key, value)` structs, four entries per row.
fn attributes() -> VortexResult<ArrayRef> {
    let entries = ROWS * 4;
    let keys = VarBinViewArray::from_iter_str((0..entries).map(|i| format!("key.{}", i % 6)));
    let values = PrimitiveArray::from_iter((0..entries).map(|i| (i * 7_919) % 300));
    let elements = StructArray::try_new(
        ["key", "value"].into(),
        vec![keys.into_array(), values.into_array()],
        entries as usize,
        Validity::NonNullable,
    )?;
    Ok(ListArray::try_new(
        elements.into_array(),
        PrimitiveArray::from_iter((0..=ROWS).map(|row| (row * 4) as u32)).into_array(),
        Validity::NonNullable,
    )?
    .into_array())
}

/// Compresses `input` with `pins` and checks it decodes to `input`.
fn replay(pins: &[PinSpec<'_>], input: &ArrayRef) -> VortexResult<PinTally> {
    let table: &'static SchemePins =
        Box::leak(Box::new(SchemePins::new(&builder().build(), pins)?));
    let compressor = builder().with_pins(table).build();
    let mut ctx = SESSION.create_execution_ctx();
    let (compressed, tally) = compressor.compress_with_tally(input, &mut ctx)?;
    assert_arrays_eq!(input, &compressed, &mut ctx);
    Ok(tally)
}

/// Every scheme the generated tables pin, and canonical, replays at the root and decodes to
/// its input.
#[rstest]
#[case::sequence(Some("vortex.int.sequence"), sequence())]
#[case::runend(Some("vortex.int.runend"), runs())]
#[case::frame_of_reference(Some("vortex.int.for"), offset_ints())]
#[case::bitpacking(Some("vortex.int.bitpacking"), small_ints())]
#[case::dict(Some("vortex.int.dict"), few_values())]
#[case::temporal(Some("vortex.ext.temporal"), timestamps())]
#[cfg_attr(feature = "zstd", case::zstd(Some("vortex.string.zstd"), strings()))]
#[case::canonical(None, few_values())]
fn pinned_scheme_decodes_to_its_input(
    #[case] decision: Option<&str>,
    #[case] input: ArrayRef,
) -> VortexResult<()> {
    let tally = replay(
        &[PinSpec {
            path: &[],
            history: &[],
            decision,
        }],
        &input,
    )?;
    assert!(tally.pinned >= 1, "{decision:?}: {tally:?}");
    assert_eq!(
        (tally.declined, tally.ineligible, tally.failed),
        (0, 0, 0),
        "{decision:?}: {tally:?}"
    );
    Ok(())
}

/// Pinning every site the search decided, with the search's decision, replays every site: none
/// is searched, so the observed keys are the lookup keys, including below struct fields, list
/// elements, extension storage and cascades.
#[test]
fn observed_sites_are_the_lookup_keys() -> VortexResult<()> {
    let input = StructArray::from_fields(&[
        ("row", sequence()),
        ("runs", runs()),
        ("offset", offset_ints()),
        ("small", small_ints()),
        ("few", few_values()),
        ("time", timestamps()),
        ("name", strings()),
        ("attributes", attributes()?),
    ])?
    .into_array();
    let mut ctx = SESSION.create_execution_ctx();
    let (_, observed) = builder().build().compress_observed(&input, &mut ctx)?;
    let mut decisions = BTreeMap::<SiteKey, Option<String>>::new();
    for site in &observed {
        let decision = site.decision.map(|id| id.to_string());
        let first = decisions
            .entry(site.key)
            .or_insert_with(|| decision.clone());
        assert_eq!(
            *first, decision,
            "one decision per site in one array: {}",
            site.key
        );
    }
    assert!(
        decisions
            .keys()
            .any(|key| key.path().steps().is_some_and(|p| p.len() >= 3)),
        "the input reaches nested sites: {decisions:?}"
    );

    let paths: Vec<_> = decisions.keys().map(SiteKey::path).collect();
    let pins: Vec<PinSpec<'_>> = decisions
        .iter()
        .zip(&paths)
        .map(|((key, decision), path)| PinSpec {
            path: path.steps().expect("observed sites fit"),
            history: key.history(),
            decision: decision.as_deref(),
        })
        .collect();
    let tally = replay(&pins, &input)?;
    assert_eq!(
        tally,
        PinTally {
            pinned: u32::try_from(observed.len()).expect("few sites"),
            ..PinTally::default()
        }
    );
    Ok(())
}
