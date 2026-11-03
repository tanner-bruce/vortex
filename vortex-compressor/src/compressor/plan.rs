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
//!
//! A replay that fails, by an error or a panic in the replayed winner's estimate or compress or
//! in any of its children, falls back to a full selection for that array. Fallbacks are counted
//! apart from declined replays (misses), with the first message per scheme kept.
//!
//! A seeded plan ([`SchemePlan::seeded`]) records nothing: its decisions come from the input
//! parts' own encoding trees through [`SchemePlan::seed`], and a site without one gets a full
//! selection on every array, counted as unseeded.
//!
//! ZY-615 cells 4: a seeded plan may also hold race sites ([`SchemePlan::race`]). At a race site
//! without a seeded decision, every candidate (the inputs' top two decisions there, and Zstd at a
//! string site) compresses the array, children included, and the smallest result is kept; ties
//! keep the earlier candidate. A race where no candidate succeeds gets a full selection.

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeMap;
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
pub mod site {
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
    /// Replay fallbacks on this thread, to tell which top-level arrays had one.
    static FALLBACKS: Cell<u64> = const { Cell::new(0) };
    /// Whether this thread is inside a replay attempt, whose panics are caught and reported once
    /// per scheme rather than by the panic hook.
    static IN_REPLAY: Cell<u32> = const { Cell::new(0) };
}

/// Whether the array being compressed on this thread is a top-level one (no site pushed).
pub(super) fn is_top_level() -> bool {
    SITE.with_borrow(Vec::is_empty)
}

/// This thread's replay fallback count so far.
pub(super) fn thread_fallbacks() -> u64 {
    FALLBACKS.get()
}

/// Runs `f`, a replay attempt, catching a panic as an error message. While it runs, the panic
/// hook stays quiet on this thread: the fallback reports the first message per scheme instead.
pub(super) fn catch_replay<R>(f: impl FnOnce() -> R) -> Result<R, String> {
    static QUIET_HOOK: std::sync::Once = std::sync::Once::new();
    QUIET_HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if IN_REPLAY.get() == 0 {
                previous(info);
            }
        }));
    });
    IN_REPLAY.set(IN_REPLAY.get() + 1);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    IN_REPLAY.set(IN_REPLAY.get() - 1);
    result.map_err(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".to_owned());
        format!("panic: {message}")
    })
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
    /// Compress with each candidate (`None` for canonical) and keep the smallest; the `usize` is
    /// the candidate's rank: 0 and 1 for the inputs' top two, 2 for the added Zstd.
    Race(Vec<(Option<&'static dyn Scheme>, usize)>),
}

/// A race site's candidates: the inputs' top decisions and an added scheme, each with its rank.
#[derive(Debug, Clone)]
struct Race {
    candidates: Vec<(Option<SchemeId>, usize)>,
}

/// A plan's counters; see [`SchemePlan::counts`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct PlanCounts {
    pub recorded: u64,
    pub replayed: u64,
    pub missed: u64,
    pub fallbacks: u64,
    pub chunks: u64,
    pub fallback_chunks: u64,
    pub unseeded: u64,
    /// Race sites seeded.
    pub race_sites: u64,
    /// Lookups that raced.
    pub raced: u64,
    /// Candidates compressed by races.
    pub race_candidates: u64,
    /// Races won by the inputs' top decision, their second, and the added Zstd.
    pub race_wins: [u64; 3],
    /// Race candidates whose estimate declined the array.
    pub race_declined: u64,
    /// Race candidates that failed by an error or a panic.
    pub race_errors: u64,
    /// Races no candidate won, which then got a full selection.
    pub race_failed: u64,
}

/// Records one decision per compression site and replays it. Shared by every array compressed
/// with the compressor that holds it; the first selection to finish at a site wins.
#[derive(Debug, Default)]
pub struct SchemePlan {
    decisions: Mutex<FxHashMap<PlanKey, Decision>>,
    recorded: AtomicU64,
    replayed: AtomicU64,
    missed: AtomicU64,
    fallbacks: AtomicU64,
    chunks: AtomicU64,
    fallback_chunks: AtomicU64,
    unseeded: AtomicU64,
    first_errors: Mutex<BTreeMap<String, String>>,
    seeded_only: bool,
    races: Mutex<FxHashMap<PlanKey, Race>>,
    raced: AtomicU64,
    race_candidates: AtomicU64,
    race_wins: [AtomicU64; 3],
    race_declined: AtomicU64,
    race_errors: AtomicU64,
    race_failed: AtomicU64,
}

impl SchemePlan {
    /// An empty plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty plan that only replays what [`SchemePlan::seed`] gives it.
    pub fn seeded() -> Self {
        Self {
            seeded_only: true,
            ..Self::default()
        }
    }

    /// The cascade step of a list's offsets below the list's own site and history.
    pub fn list_offsets_step() -> (SchemeId, usize) {
        (super::ROOT_SCHEME_ID, super::structural::LIST_OFFSETS_CHILD)
    }

    /// Seeds `decision` (`None` for canonical) at the site `site` with cascade `history`. A
    /// seeded site matches either logical-strings choice.
    pub fn seed(&self, site: &[u32], history: &[(SchemeId, usize)], decision: Option<SchemeId>) {
        let key = PlanKey {
            site: site.to_vec(),
            history: history.to_vec(),
            logical_strings: false,
        };
        let decision = decision.map_or(Decision::Canonical, Decision::Scheme);
        if self.decisions.lock().insert(key, decision).is_none() {
            self.recorded.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Makes the site `site` with cascade `history` a race site, unless it has a seeded
    /// decision: `top` are the inputs' top decisions there (`None` for canonical), most rows
    /// first, and `extra` an added candidate, skipped when `top` has it.
    pub fn race(
        &self,
        site: &[u32],
        history: &[(SchemeId, usize)],
        top: &[Option<SchemeId>],
        extra: Option<SchemeId>,
    ) {
        let key = PlanKey {
            site: site.to_vec(),
            history: history.to_vec(),
            logical_strings: false,
        };
        let mut candidates = top
            .iter()
            .take(2)
            .enumerate()
            .map(|(rank, decision)| (*decision, rank))
            .collect::<Vec<_>>();
        if let Some(extra) = extra
            && !top.contains(&Some(extra))
        {
            candidates.push((Some(extra), 2));
        }
        if candidates.is_empty() {
            return;
        }
        self.races.lock().insert(key, Race { candidates });
    }

    /// Whether the site `site` with cascade `history` has a seeded decision.
    pub fn is_seeded(&self, site: &[u32], history: &[(SchemeId, usize)]) -> bool {
        let key = PlanKey {
            site: site.to_vec(),
            history: history.to_vec(),
            logical_strings: false,
        };
        self.decisions.lock().contains_key(&key)
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
        let mut key = PlanKey::new(ctx);
        if self.seeded_only {
            key.logical_strings = false;
        }
        let decision = self.decisions.lock().get(&key).copied();
        if decision.is_none()
            && self.seeded_only
            && let Some(race) = self.races.lock().get(&key)
        {
            let candidates = race
                .candidates
                .iter()
                .filter_map(|(decision, rank)| match decision {
                    None => Some((None, *rank)),
                    Some(id) => eligible
                        .iter()
                        .find(|s| s.id() == *id)
                        .map(|scheme| (Some(*scheme), *rank)),
                })
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                self.race_failed.fetch_add(1, Ordering::Relaxed);
                return Planned::Off;
            }
            self.raced.fetch_add(1, Ordering::Relaxed);
            return Planned::Race(candidates);
        }
        match decision {
            None if self.seeded_only => {
                self.unseeded.fetch_add(1, Ordering::Relaxed);
                Planned::Off
            }
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

    /// Counts a replay whose winner declined the array, which then got a full selection.
    pub(super) fn miss(&self) {
        self.replayed.fetch_sub(1, Ordering::Relaxed);
        self.missed.fetch_add(1, Ordering::Relaxed);
    }

    /// Counts a replay that failed with `message` (an error or a caught panic), which then got a
    /// full selection. The first message per scheme is kept here and logged once per process.
    pub(super) fn fallback(&self, scheme: SchemeId, message: &str) {
        static LOGGED: Mutex<Vec<String>> = Mutex::new(Vec::new());
        self.replayed.fetch_sub(1, Ordering::Relaxed);
        self.fallbacks.fetch_add(1, Ordering::Relaxed);
        FALLBACKS.set(FALLBACKS.get() + 1);
        let scheme = scheme.to_string();
        let first_line = message.lines().next().unwrap_or_default();
        self.first_errors
            .lock()
            .entry(scheme.clone())
            .or_insert_with(|| first_line.to_owned());
        let mut logged = LOGGED.lock();
        if !logged.contains(&scheme) {
            eprintln!("ZY615_REPLAY_FALLBACK scheme={scheme} error={message:?}");
            logged.push(scheme);
        }
    }

    /// Counts one race candidate compressed, and how it ended: `Some(true)` compressed,
    /// `Some(false)` declined by its estimate, `None` failed with `error` (scheme, message).
    pub(super) fn race_candidate(&self, outcome: Option<bool>, error: Option<(SchemeId, &str)>) {
        self.race_candidates.fetch_add(1, Ordering::Relaxed);
        match outcome {
            Some(true) => {}
            Some(false) => {
                self.race_declined.fetch_add(1, Ordering::Relaxed);
            }
            None => {
                self.race_errors.fetch_add(1, Ordering::Relaxed);
                if let Some((scheme, message)) = error {
                    let first_line = message.lines().next().unwrap_or_default();
                    self.first_errors
                        .lock()
                        .entry(format!("race {scheme}"))
                        .or_insert_with(|| first_line.to_owned());
                }
            }
        }
    }

    /// Counts a race won by the candidate of rank `rank`, or one no candidate won.
    pub(super) fn race_result(&self, rank: Option<usize>) {
        match rank {
            Some(rank) => {
                self.race_wins[rank.min(2)].fetch_add(1, Ordering::Relaxed);
            }
            None => {
                self.race_failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Counts one top-level array compressed with this plan, and whether any replay in it fell
    /// back.
    pub(super) fn chunk(&self, had_fallback: bool) {
        self.chunks.fetch_add(1, Ordering::Relaxed);
        if had_fallback {
            self.fallback_chunks.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Sites recorded or seeded, selections replayed, replays declined by the winner's estimate
    /// (misses), replays that failed and fell back, top-level arrays, top-level arrays with a
    /// fallback, and lookups a seeded plan had no decision for.
    pub fn counts(&self) -> PlanCounts {
        PlanCounts {
            recorded: self.recorded.load(Ordering::Relaxed),
            replayed: self.replayed.load(Ordering::Relaxed),
            missed: self.missed.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
            chunks: self.chunks.load(Ordering::Relaxed),
            fallback_chunks: self.fallback_chunks.load(Ordering::Relaxed),
            unseeded: self.unseeded.load(Ordering::Relaxed),
            race_sites: self.races.lock().len() as u64,
            raced: self.raced.load(Ordering::Relaxed),
            race_candidates: self.race_candidates.load(Ordering::Relaxed),
            race_wins: [
                self.race_wins[0].load(Ordering::Relaxed),
                self.race_wins[1].load(Ordering::Relaxed),
                self.race_wins[2].load(Ordering::Relaxed),
            ],
            race_declined: self.race_declined.load(Ordering::Relaxed),
            race_errors: self.race_errors.load(Ordering::Relaxed),
            race_failed: self.race_failed.load(Ordering::Relaxed),
        }
    }

    /// The first fallback message per scheme, by scheme id.
    pub fn first_errors(&self) -> BTreeMap<String, String> {
        self.first_errors.lock().clone()
    }

    /// One line per recorded site, `site-path | cascade | decision`, and per race site,
    /// `site-path | cascade | race candidate / candidate / +added`, sorted.
    pub fn describe(&self) -> Vec<String> {
        let mut lines = self
            .decisions
            .lock()
            .iter()
            .map(|(key, decision)| {
                let mut line = key.describe();
                match decision {
                    Decision::Canonical => line.push_str(" | canonical"),
                    Decision::Scheme(id) => {
                        let _ = write!(line, " | {id}");
                    }
                }
                line
            })
            .collect::<Vec<_>>();
        lines.extend(self.races.lock().iter().map(|(key, race)| {
            let mut line = key.describe();
            line.push_str(" | race");
            for (i, (decision, rank)) in race.candidates.iter().enumerate() {
                line.push_str(if i > 0 { " / " } else { " " });
                if *rank == 2 {
                    line.push('+');
                }
                match decision {
                    None => line.push_str("canonical"),
                    Some(id) => {
                        let _ = write!(line, "{id}");
                    }
                }
            }
            line
        }));
        lines.sort();
        lines
    }
}

impl PlanKey {
    /// `site-path | cascade`, with ` (logical strings)` when set.
    fn describe(&self) -> String {
        let mut line = String::new();
        for (i, index) in self.site.iter().enumerate() {
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
        for (i, (id, child)) in self.history.iter().enumerate() {
            if i > 0 {
                line.push_str(" > ");
            }
            let _ = write!(line, "{id}[{child}]");
        }
        if self.logical_strings {
            line.push_str(" (logical strings)");
        }
        line
    }
}
