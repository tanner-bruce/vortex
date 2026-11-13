// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheme pins: a fixed decision per compression site, replayed instead of searched.
//!
//! A [`SchemePins`] table maps a site within one top-level array ([`SiteKey`]: its structural
//! path and cascade history) to a scheme or to canonical. A compressor with pins
//! ([`CascadingCompressor::with_pins`]) looks each site up after the empty, all-null and constant
//! checks. Without a pin, or with a pinned scheme that is not eligible at the site, it searches as
//! usual. Otherwise the pinned scheme must still admit the array by its own estimate, without
//! sampling; it then compresses with only its own statistics, and its children look up their own
//! sites. A declined or failed replay searches. Sample compression never reads pins.
//!
//! Pins change which scheme a site uses, never the values: every pinned array decodes to the
//! values the search would have written.

use std::cell::Cell;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use super::CascadingCompressor;
use super::ROOT_SCHEME_ID;
use crate::scheme::CompressorContext;
use crate::scheme::MAX_CASCADE;
use crate::scheme::ROOT_SCHEME_NAME;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;
use crate::scheme::SiteKey;
use crate::scheme::SitePath;

/// One pin as a static table lists it, before [`SchemePins::new`] resolves it.
#[derive(Debug, Clone, Copy)]
pub struct PinSpec<'a> {
    /// The structural steps from the top-level array (see [`site`](crate::scheme::site)).
    pub path: &'a [u32],
    /// The cascade history from the top, as `(scheme name, child index)` pairs; the compressor's
    /// own list cascading is [`ROOT_SCHEME_NAME`].
    pub history: &'a [(&'a str, u32)],
    /// The pinned scheme's name, or `None` to leave the array canonical.
    pub decision: Option<&'a str>,
}

/// An immutable table of pinned decisions, resolved against one compressor's scheme set.
///
/// Sites are fixed-size keys in a sorted array, so a lookup allocates nothing.
#[derive(Debug)]
pub struct SchemePins {
    /// Sorted by key; `None` pins canonical.
    sites: Box<[(SiteKey, Option<&'static dyn Scheme>)]>,
}

impl SchemePins {
    /// Resolves `pins` against `compressor`'s schemes.
    ///
    /// # Errors
    ///
    /// Fails if a pin names a scheme `compressor` does not have, has a path deeper than
    /// [`MAX_SITE_DEPTH`](crate::scheme::MAX_SITE_DEPTH) or a history deeper than
    /// [`MAX_CASCADE`], or two pins share a site.
    pub fn new(compressor: &CascadingCompressor, pins: &[PinSpec<'_>]) -> VortexResult<Self> {
        let resolve = |name: &str| {
            compressor
                .schemes
                .iter()
                .copied()
                .find(|scheme| scheme.id().name == name)
                .ok_or_else(|| vortex_err!("scheme pin names unknown scheme {name}"))
        };
        let mut sites = Vec::with_capacity(pins.len());
        for pin in pins {
            let path = SitePath::new(pin.path).ok_or_else(|| {
                vortex_err!("scheme pin path {:?} is deeper than a site path", pin.path)
            })?;
            if pin.history.len() > MAX_CASCADE {
                vortex_bail!(
                    "scheme pin history {:?} is deeper than MAX_CASCADE",
                    pin.history
                );
            }
            let mut history = [(ROOT_SCHEME_ID, 0); MAX_CASCADE];
            for (entry, &(name, child)) in history.iter_mut().zip(pin.history) {
                let id = if name == ROOT_SCHEME_NAME {
                    ROOT_SCHEME_ID
                } else {
                    resolve(name)?.id()
                };
                *entry = (id, child as usize);
            }
            let key = SiteKey::new(path, &history[..pin.history.len()])
                .ok_or_else(|| vortex_err!("scheme pin {pin:?} has no site key"))?;
            sites.push((key, pin.decision.map(resolve).transpose()?));
        }
        sites.sort_unstable_by_key(|(key, _)| *key);
        if let Some(pair) = sites.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            vortex_bail!("two scheme pins share the site {}", pair[0].0);
        }
        Ok(Self {
            sites: sites.into_boxed_slice(),
        })
    }

    /// The number of pinned sites.
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// Whether the table pins no site.
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// The pinned sites and their decisions (`None` for canonical), in key order.
    pub fn iter(&self) -> impl Iterator<Item = (SiteKey, Option<SchemeId>)> + '_ {
        self.sites
            .iter()
            .map(|(key, decision)| (*key, decision.map(|scheme| scheme.id())))
    }

    /// The decision pinned at `ctx`'s site: `Some(None)` for canonical, `None` without a pin.
    pub(super) fn get(&self, ctx: &CompressorContext) -> Option<Option<&'static dyn Scheme>> {
        let key = SiteKey::new(ctx.site(), ctx.cascade_history())?;
        self.sites
            .binary_search_by(|(site, _)| site.cmp(&key))
            .ok()
            .map(|index| self.sites[index].1)
    }
}

/// Pin lookups during one [`CascadingCompressor::compress_with_tally`] call, by outcome.
///
/// Every site of a compressor with pins counts once, except empty, all-null and constant arrays,
/// which finish before the lookup, and sites inside a failed replay, which count only as its
/// failure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PinTally {
    /// Sites compressed with their pinned decision.
    pub pinned: u32,
    /// Sites without a pin, searched.
    pub searched: u32,
    /// Pinned schemes that declined the array by their own estimate; the site was searched.
    pub declined: u32,
    /// Pinned schemes not eligible at the site; the site was searched.
    pub ineligible: u32,
    /// Pinned schemes that returned an error; the site was searched.
    pub failed: u32,
    /// The scheme of the first failure.
    pub first_failure: Option<SchemeId>,
}

impl PinTally {
    /// No lookups, for the thread's running tally.
    const ZERO: Self = Self {
        pinned: 0,
        searched: 0,
        declined: 0,
        ineligible: 0,
        failed: 0,
        first_failure: None,
    };

    /// Adds `other`'s counts; the first failure stays the earlier one.
    pub fn add(&mut self, other: &Self) {
        self.pinned = self.pinned.saturating_add(other.pinned);
        self.searched = self.searched.saturating_add(other.searched);
        self.declined = self.declined.saturating_add(other.declined);
        self.ineligible = self.ineligible.saturating_add(other.ineligible);
        self.failed = self.failed.saturating_add(other.failed);
        self.first_failure = self.first_failure.or(other.first_failure);
    }
}

/// The outcome of one counted lookup.
#[derive(Debug, Clone, Copy)]
pub(super) enum Lookup {
    /// Compressed with the pinned decision.
    Pinned,
    /// No pin; searched.
    Searched,
    /// The pinned scheme declined; searched.
    Declined,
    /// The pinned scheme is not eligible here; searched.
    Ineligible,
}

thread_local! {
    /// This thread's tally. One `compress` call runs on one thread; the style guide allows a
    /// thread-local for diagnostics that never change results.
    static TALLY: Cell<PinTally> = const { Cell::new(PinTally::ZERO) };
}

/// Counts one lookup on this thread.
pub(super) fn count(lookup: Lookup) {
    TALLY.with(|tally| {
        let mut counts = tally.get();
        let count = match lookup {
            Lookup::Pinned => &mut counts.pinned,
            Lookup::Searched => &mut counts.searched,
            Lookup::Declined => &mut counts.declined,
            Lookup::Ineligible => &mut counts.ineligible,
        };
        *count = count.saturating_add(1);
        tally.set(counts);
    });
}

/// Replaces this thread's tally with `tally`, returning the previous one.
pub(super) fn replace(tally: PinTally) -> PinTally {
    TALLY.with(|cell| cell.replace(tally))
}

/// Restores `before`, the tally from before a replay, and counts the replay of `scheme` failed.
pub(super) fn fail(before: PinTally, scheme: SchemeId) {
    TALLY.with(|tally| {
        let mut counts = before;
        counts.failed = counts.failed.saturating_add(1);
        counts.first_failure = counts.first_failure.or(Some(scheme));
        tally.set(counts);
    });
}

/// This thread's tally so far.
pub(super) fn current() -> PinTally {
    TALLY.with(Cell::get)
}

/// A site the search decided, as the `site-observer` feature reports it.
#[cfg(feature = "site-observer")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservedSite {
    /// The site, keyed as a pin lookup keys it.
    pub key: SiteKey,
    /// The search's choice, or `None` for canonical.
    pub decision: Option<SchemeId>,
}

#[cfg(feature = "site-observer")]
thread_local! {
    /// The sites observed during one `compress_observed` call on this thread.
    static OBSERVED: std::cell::RefCell<Option<Vec<ObservedSite>>> =
        const { std::cell::RefCell::new(None) };
}

/// Reports the search's decision at `ctx`'s site to an active observer.
#[cfg(feature = "site-observer")]
pub(super) fn observe(ctx: &CompressorContext, decision: Option<SchemeId>) {
    if ctx.is_sample() {
        return;
    }
    let Some(key) = SiteKey::new(ctx.site(), ctx.cascade_history()) else {
        return;
    };
    OBSERVED.with_borrow_mut(|observed| {
        if let Some(observed) = observed {
            observed.push(ObservedSite { key, decision });
        }
    });
}

/// Without the `site-observer` feature, observing does nothing.
#[cfg(not(feature = "site-observer"))]
#[inline(always)]
pub(super) fn observe(_ctx: &CompressorContext, _decision: Option<SchemeId>) {}

/// Starts observing on this thread, returning the previous observer's sites.
#[cfg(feature = "site-observer")]
pub(super) fn start_observing() -> Option<Vec<ObservedSite>> {
    OBSERVED.with_borrow_mut(|observed| observed.replace(Vec::new()))
}

/// Stops observing, restoring `previous`, and returns the sites observed.
#[cfg(feature = "site-observer")]
pub(super) fn stop_observing(previous: Option<Vec<ObservedSite>>) -> Vec<ObservedSite> {
    OBSERVED
        .with_borrow_mut(|observed| std::mem::replace(observed, previous))
        .unwrap_or_default()
}
