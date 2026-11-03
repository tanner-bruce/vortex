// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ZY-615 phase 0 measurement only, never merged: a scheme plan that records the winner of the
//! first full scheme selection at each compression site and replays it for every later array
//! compressed with the same plan, so sampling runs once per site instead of once per chunk.
//!
//! A site is the structural path from the top-level array (struct field, list elements, ...)
//! plus the cascade history at that point, so a replayed winner's own children find their
//! recorded winners too and the whole cascade is fixed. Sample compression never reads or writes
//! the plan. Empty, all-null and constant arrays are still detected per array, before the plan.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::scheme::CompressorContext;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;

/// Structural child markers for the site path; struct fields and union children use their index.
pub(super) mod site {
    /// The elements of a list, list view or fixed-size list.
    pub const LIST_ELEMENTS: u32 = u32::MAX;
    /// The storage of an extension array.
    pub const EXTENSION_STORAGE: u32 = u32::MAX - 1;
    /// The type ids of a union array.
    pub const UNION_TYPE_IDS: u32 = u32::MAX - 2;
    /// The child slots of a variant array start here.
    pub const VARIANT_SLOTS: u32 = u32::MAX - 64;
}

thread_local! {
    /// The structural path of the array being compressed on this thread. Compression of one
    /// array runs synchronously on one thread, so a stack is enough.
    static SITE: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` with `index` pushed on this thread's site path.
pub(super) fn with_site<R>(index: u32, f: impl FnOnce() -> R) -> R {
    struct Pop;
    impl Drop for Pop {
        fn drop(&mut self) {
            SITE.with_borrow_mut(|path| {
                path.pop();
            });
        }
    }
    SITE.with_borrow_mut(|path| path.push(index));
    let _pop = Pop;
    f()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct PlanKey {
    site: Vec<u32>,
    history: Vec<(SchemeId, usize)>,
    logical_strings: bool,
}

impl PlanKey {
    pub(super) fn new(ctx: &CompressorContext) -> Self {
        Self {
            site: SITE.with_borrow(Clone::clone),
            history: ctx.cascade_history().to_vec(),
            logical_strings: ctx.logical_strings,
        }
    }
}

/// What a site's first full selection chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Decision {
    /// No scheme beat the canonical encoding.
    Canonical,
    /// This scheme won.
    Scheme(SchemeId),
}

/// The plan's view of one site for one array.
pub(super) enum Planned {
    /// No plan, or a sample: select as usual and record nothing.
    Off,
    /// The site has no decision yet: select as usual and record the result.
    Record(PlanKey),
    /// Replay this decision without selecting.
    Replay(Option<&'static dyn Scheme>),
}

/// Records one decision per compression site and replays it. Shared by every array compressed
/// with the compressor that holds it; the first selection to finish at a site wins.
#[derive(Debug, Default)]
pub struct SchemePlan {
    decisions: Mutex<FxHashMap<PlanKey, Decision>>,
    recorded: AtomicU64,
    replayed: AtomicU64,
    missed: AtomicU64,
}

impl SchemePlan {
    /// An empty plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// Looks up the site described by `ctx` and this thread's site path.
    pub(super) fn lookup(
        &self,
        ctx: &CompressorContext,
        eligible: &[&'static dyn Scheme],
    ) -> Planned {
        if ctx.is_sample() {
            return Planned::Off;
        }
        let key = PlanKey::new(ctx);
        let decision = self.decisions.lock().get(&key).copied();
        match decision {
            None => Planned::Record(key),
            Some(Decision::Canonical) => {
                self.replayed.fetch_add(1, Ordering::Relaxed);
                Planned::Replay(None)
            }
            Some(Decision::Scheme(id)) => match eligible.iter().find(|s| s.id() == id) {
                Some(scheme) => {
                    self.replayed.fetch_add(1, Ordering::Relaxed);
                    Planned::Replay(Some(*scheme))
                }
                // The recorded winner is not eligible for this array (it should not happen for
                // one column's chunks): select as usual and keep the recorded decision.
                None => {
                    self.missed.fetch_add(1, Ordering::Relaxed);
                    Planned::Off
                }
            },
        }
    }

    /// Records the selection made at `key`, unless another array recorded the site first.
    pub(super) fn record(&self, key: PlanKey, winner: Option<SchemeId>) {
        let decision = winner.map_or(Decision::Canonical, Decision::Scheme);
        let mut decisions = self.decisions.lock();
        if !decisions.contains_key(&key) {
            decisions.insert(key, decision);
            self.recorded.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Sites recorded, selections replayed, and replays that fell back to a full selection.
    pub fn counts(&self) -> (u64, u64, u64) {
        (
            self.recorded.load(Ordering::Relaxed),
            self.replayed.load(Ordering::Relaxed),
            self.missed.load(Ordering::Relaxed),
        )
    }

    /// One line per recorded site, `site-path | cascade | decision`, sorted.
    pub fn describe(&self) -> Vec<String> {
        let mut lines = self
            .decisions
            .lock()
            .iter()
            .map(|(key, decision)| {
                let mut line = String::new();
                for (i, index) in key.site.iter().enumerate() {
                    if i > 0 {
                        line.push('/');
                    }
                    match *index {
                        site::LIST_ELEMENTS => line.push_str("elements"),
                        site::EXTENSION_STORAGE => line.push_str("storage"),
                        site::UNION_TYPE_IDS => line.push_str("type_ids"),
                        other => {
                            let _ = write!(line, "{other}");
                        }
                    }
                }
                line.push_str(" | ");
                for (i, (id, child)) in key.history.iter().enumerate() {
                    if i > 0 {
                        line.push_str(" > ");
                    }
                    let _ = write!(line, "{id}[{child}]");
                }
                if key.logical_strings {
                    line.push_str(" (logical strings)");
                }
                match decision {
                    Decision::Canonical => line.push_str(" | canonical"),
                    Decision::Scheme(id) => {
                        let _ = write!(line, " | {id}");
                    }
                }
                line
            })
            .collect::<Vec<_>>();
        lines.sort();
        lines
    }
}
