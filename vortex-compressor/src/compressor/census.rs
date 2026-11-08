// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer leaf census for measurement builds (the `leaf-census` feature). Not for production
//! builds: it counts into process-wide atomics and a locked list.

use std::cell::Cell;
use std::fmt::Write;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

/// Where a non-sample integer leaf went after the constant check.
#[derive(Debug, Clone, Copy)]
pub(super) enum Group {
    /// Finished by the constant check.
    Constant,
    /// Not constant, replayed by its pinned scheme (including replays that then decline).
    Replayed,
    /// Not constant, its pinned scheme declined it; it was searched afterwards (and counts again
    /// under `Searched`).
    Declined,
    /// Not constant, pinned canonical.
    Canonical,
    /// Not constant, no pin; searched.
    Unpinned,
    /// Not constant, its pinned scheme not eligible; searched.
    Ineligible,
    /// Not constant, searched without a counted lookup: no pins, or after a declined or failed
    /// replay.
    Searched,
}

const GROUPS: [(Group, &str); 7] = [
    (Group::Constant, "constant"),
    (Group::Replayed, "replayed"),
    (Group::Declined, "declined"),
    (Group::Canonical, "canonical"),
    (Group::Unpinned, "unpinned"),
    (Group::Ineligible, "ineligible"),
    (Group::Searched, "searched"),
];

/// Leaves and values per group.
static LEAVES: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
static VALUES: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// Full `IntegerStats` generated during a replay: scheme, distinct counting, leaves and values.
static FULL_STATS: Mutex<Vec<(&'static str, bool, u64, u64)>> = Mutex::new(Vec::new());

thread_local! {
    /// The pinned scheme being replayed on this thread, `None` during a search.
    static REPLAYING: Cell<Option<&'static str>> = const { Cell::new(None) };
}

/// Counts one integer leaf of `len` values in `group`.
pub(super) fn count(group: Group, len: usize) {
    let index = group as usize;
    LEAVES[index].fetch_add(1, Ordering::Relaxed);
    VALUES[index].fetch_add(u64::try_from(len).unwrap_or(u64::MAX), Ordering::Relaxed);
}

/// Sets the scheme being replayed (`None` for a search) until the guard drops.
pub(super) fn replaying(scheme: Option<&'static str>) -> ReplayGuard {
    ReplayGuard(REPLAYING.with(|cell| cell.replace(scheme)))
}

/// Restores the previous replay scheme when dropped.
pub(super) struct ReplayGuard(Option<&'static str>);

impl Drop for ReplayGuard {
    fn drop(&mut self) {
        REPLAYING.with(|cell| cell.set(self.0));
    }
}

/// Records one full `IntegerStats` generation of `len` values, if it happens during a replay.
pub(crate) fn full_integer_stats(count_distinct_values: bool, len: usize) {
    let Some(scheme) = REPLAYING.with(Cell::get) else {
        return;
    };
    let len = u64::try_from(len).unwrap_or(u64::MAX);
    let mut full = FULL_STATS.lock();
    match full
        .iter_mut()
        .find(|(name, distinct, ..)| *name == scheme && *distinct == count_distinct_values)
    {
        Some(entry) => {
            entry.2 += 1;
            entry.3 += len;
        }
        None => full.push((scheme, count_distinct_values, 1, len)),
    }
}

/// Returns the census so far as one line, and resets it.
pub fn take_leaf_census() -> String {
    let mut line = String::new();
    for (group, name) in GROUPS {
        let index = group as usize;
        let leaves = LEAVES[index].swap(0, Ordering::Relaxed);
        let values = VALUES[index].swap(0, Ordering::Relaxed);
        let _ = write!(line, "{name}={leaves}/{values} ");
    }
    let mut full = std::mem::take(&mut *FULL_STATS.lock());
    full.sort_unstable();
    line.push_str("full_stats_in_replay=[");
    for (scheme, distinct, leaves, values) in full {
        let _ = write!(line, "{scheme}:distinct={distinct}:{leaves}/{values} ");
    }
    line.push(']');
    line
}
