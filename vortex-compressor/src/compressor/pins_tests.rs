// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheme pin lookups, fallbacks and tallies, with test schemes that record their compressions.

use std::cell::RefCell;
use std::sync::LazyLock;

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use super::CascadingCompressor;
use super::PinSpec;
use super::PinTally;
use super::SchemePins;
use crate::builtins::IntDictScheme;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::EstimateVerdict;
use crate::scheme::MAX_SITE_DEPTH;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SitePath;
use crate::scheme::site;
use crate::stats::ArrayAndStats;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

thread_local! {
    /// The test schemes compressed on this thread, in order.
    static CALLS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// A scheme over integer (or, with `floats`, float) primitives with a fixed ratio estimate
/// (`None` skips) whose compression returns its input, or fails.
#[derive(Debug)]
struct PinTestScheme {
    name: &'static str,
    ratio: Option<f64>,
    floats: bool,
    fails: bool,
}

static ADMIT: PinTestScheme = PinTestScheme {
    name: "test.pin.admit",
    ratio: Some(2.0),
    floats: false,
    fails: false,
};
static BETTER: PinTestScheme = PinTestScheme {
    name: "test.pin.better",
    ratio: Some(3.0),
    floats: false,
    fails: false,
};
static DECLINE: PinTestScheme = PinTestScheme {
    name: "test.pin.decline",
    ratio: None,
    floats: false,
    fails: false,
};
static FAIL: PinTestScheme = PinTestScheme {
    name: "test.pin.fail",
    ratio: Some(1.5),
    floats: false,
    fails: true,
};
static FLOATS: PinTestScheme = PinTestScheme {
    name: "test.pin.floats",
    ratio: Some(2.0),
    floats: true,
    fails: false,
};

impl Scheme for PinTestScheme {
    fn scheme_name(&self) -> &'static str {
        self.name
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        matches!(
            canonical,
            Canonical::Primitive(primitive)
                if if self.floats { primitive.ptype().is_float() } else { primitive.ptype().is_int() }
        )
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        Vec::new()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(match self.ratio {
            Some(ratio) => EstimateVerdict::Ratio(ratio),
            None => EstimateVerdict::Skip,
        })
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        CALLS.with_borrow_mut(|calls| calls.push(self.name));
        if self.fails {
            vortex_bail!("{} fails by design", self.name);
        }
        Ok(data.array().clone())
    }
}

/// 4,096 integers with five distinct values, none constant.
fn ints() -> ArrayRef {
    PrimitiveArray::from_iter((0..4096_i64).map(|i| (i % 5) * 1_000_003)).into_array()
}

fn pin<'a>(
    path: &'a [u32],
    history: &'a [(&'a str, u32)],
    decision: Option<&'a str>,
) -> PinSpec<'a> {
    PinSpec {
        path,
        history,
        decision,
    }
}

/// `pins` resolved against `compressor` and leaked, and the compressor with them.
fn pinned(compressor: CascadingCompressor, pins: &[PinSpec<'_>]) -> CascadingCompressor {
    let table: &'static SchemePins = Box::leak(Box::new(
        SchemePins::new(&compressor, pins).expect("pins resolve"),
    ));
    compressor.with_pins(table)
}

/// Compresses `input` with `compressor`, checks it decodes to `input`, and returns the tally and
/// the test schemes compressed.
fn run(compressor: &CascadingCompressor, input: &ArrayRef) -> (PinTally, Vec<&'static str>) {
    let mut ctx = SESSION.create_execution_ctx();
    CALLS.with_borrow_mut(Vec::clear);
    let (compressed, tally) = compressor
        .compress_with_tally(input, &mut ctx)
        .expect("compress");
    assert_arrays_eq!(input, &compressed, &mut ctx);
    (tally, CALLS.with_borrow_mut(std::mem::take))
}

#[test]
fn pinned_scheme_replaces_the_search() {
    let compressor = pinned(
        CascadingCompressor::new(vec![&ADMIT, &BETTER]),
        &[pin(&[], &[], Some(ADMIT.name))],
    );
    let (tally, calls) = run(&compressor, &ints());
    assert_eq!(
        calls,
        [ADMIT.name],
        "the search would choose {}",
        BETTER.name
    );
    assert_eq!(
        tally,
        PinTally {
            pinned: 1,
            ..PinTally::default()
        }
    );
}

#[test]
fn canonical_pin_compresses_nothing() {
    let compressor = pinned(
        CascadingCompressor::new(vec![&ADMIT]),
        &[pin(&[], &[], None)],
    );
    let (tally, calls) = run(&compressor, &ints());
    assert!(calls.is_empty());
    assert_eq!(tally.pinned, 1);
    assert_eq!(tally.searched, 0);
}

#[test]
fn site_without_a_pin_searches() {
    let compressor = pinned(
        CascadingCompressor::new(vec![&ADMIT, &BETTER]),
        &[pin(&[3], &[], Some(ADMIT.name))],
    );
    let (tally, calls) = run(&compressor, &ints());
    assert_eq!(calls, [BETTER.name]);
    assert_eq!(
        tally,
        PinTally {
            searched: 1,
            ..PinTally::default()
        }
    );
}

#[test]
fn declined_pin_searches() {
    let compressor = pinned(
        CascadingCompressor::new(vec![&DECLINE, &ADMIT]),
        &[pin(&[], &[], Some(DECLINE.name))],
    );
    let (tally, calls) = run(&compressor, &ints());
    assert_eq!(calls, [ADMIT.name]);
    assert_eq!(
        tally,
        PinTally {
            declined: 1,
            ..PinTally::default()
        }
    );
}

#[test]
fn ineligible_pin_searches() {
    let compressor = pinned(
        CascadingCompressor::new(vec![&FLOATS, &ADMIT]),
        &[pin(&[], &[], Some(FLOATS.name))],
    );
    let (tally, calls) = run(&compressor, &ints());
    assert_eq!(calls, [ADMIT.name]);
    assert_eq!(
        tally,
        PinTally {
            ineligible: 1,
            ..PinTally::default()
        }
    );
}

#[test]
fn failed_pin_searches_and_keeps_the_scheme() {
    let compressor = pinned(
        CascadingCompressor::new(vec![&FAIL, &ADMIT]),
        &[pin(&[], &[], Some(FAIL.name))],
    );
    let (tally, calls) = run(&compressor, &ints());
    assert_eq!(calls, [FAIL.name, ADMIT.name]);
    assert_eq!(
        tally,
        PinTally {
            failed: 1,
            first_failure: Some(FAIL.id()),
            ..PinTally::default()
        }
    );
}

#[test]
fn children_of_a_pinned_scheme_look_up_their_own_sites() {
    let dict = IntDictScheme.id().to_string();
    let history = [(dict.as_str(), 1)];
    let compressor = pinned(
        CascadingCompressor::new(vec![&IntDictScheme, &ADMIT]),
        &[pin(&[], &[], Some(dict.as_str())), pin(&[], &history, None)],
    );
    let (tally, calls) = run(&compressor, &ints());
    // The values (child 0) have no pin and search; the codes (child 1) stay canonical.
    assert_eq!(calls, [ADMIT.name]);
    assert_eq!(
        tally,
        PinTally {
            pinned: 2,
            searched: 1,
            ..PinTally::default()
        }
    );
}

#[test]
fn struct_fields_have_their_own_sites() -> VortexResult<()> {
    let compressor = pinned(
        CascadingCompressor::new(vec![&ADMIT, &BETTER]),
        &[pin(&[1], &[], Some(ADMIT.name))],
    );
    let input = StructArray::try_new(
        ["a", "b"].into(),
        vec![ints(), ints()],
        4096,
        Validity::NonNullable,
    )?
    .into_array();
    let (tally, calls) = run(&compressor, &input);
    assert_eq!(calls, [BETTER.name, ADMIT.name]);
    assert_eq!(
        tally,
        PinTally {
            pinned: 1,
            searched: 1,
            ..PinTally::default()
        }
    );
    Ok(())
}

#[test]
fn no_pins_tally_nothing() {
    let compressor = CascadingCompressor::new(vec![&ADMIT]);
    let (tally, calls) = run(&compressor, &ints());
    assert_eq!(calls, [ADMIT.name]);
    assert_eq!(tally, PinTally::default());
}

#[test]
fn unknown_or_malformed_pins_fail_at_construction() {
    let compressor = CascadingCompressor::new(vec![&ADMIT]);
    let deep = [0; MAX_SITE_DEPTH + 1];
    let cases: [&[PinSpec<'_>]; 5] = [
        &[pin(&[], &[], Some("test.pin.missing"))],
        &[pin(&[], &[("test.pin.missing", 0)], None)],
        &[pin(&deep, &[], None)],
        &[pin(&[], &[("test.pin.admit", 0); 4], None)],
        &[pin(&[2], &[], None), pin(&[2], &[], Some(ADMIT.name))],
    ];
    for pins in cases {
        assert!(SchemePins::new(&compressor, pins).is_err(), "{pins:?}");
    }
    let table = SchemePins::new(
        &compressor,
        &[
            pin(
                &[site::LIST_ELEMENTS, 0],
                &[("vortex.compressor.root", 1)],
                None,
            ),
            pin(&[], &[], Some(ADMIT.name)),
        ],
    )
    .expect("valid pins");
    assert_eq!(table.len(), 2);
}

#[test]
fn site_paths_overflow_past_the_depth() {
    let mut path = SitePath::ROOT;
    for step in 0..MAX_SITE_DEPTH {
        path = path.child_index(step);
        assert!(!path.is_overflowed());
    }
    assert!(path.child(site::LIST_ELEMENTS).is_overflowed());
    assert!(
        SitePath::ROOT
            .child_index(site::FIRST_MARKER as usize)
            .is_overflowed()
    );
    let shown = SitePath::new(&[2, site::LIST_ELEMENTS, 0, site::EXTENSION_STORAGE])
        .expect("fits")
        .to_string();
    assert_eq!(shown, "2/elements/0/storage");
}
