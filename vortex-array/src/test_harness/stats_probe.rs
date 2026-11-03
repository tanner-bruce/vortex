// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A process-wide probe for work done on arrays while it is active.
//!
//! While a probe is [started](start), every statistic computed rather than read from the cache,
//! every aggregate accumulation and every encoding execution (any decode step) is recorded with
//! the array it ran on. A test brackets one operation with [`start`] and [`stop`] to show that the
//! operation computed no statistic and decoded nothing.
//!
//! The probe is global because write paths run work on CPU tasks on other threads. Only one test
//! in a process may use it at a time: put a probing test in its own test binary.

use std::fmt::Display;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;

use crate::ArrayRef;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static EVENTS: Mutex<Vec<StatsProbeEvent>> = Mutex::new(Vec::new());

/// What the probe saw being done to an array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatsProbeKind {
    /// A statistic computed because the cache held no exact value.
    ComputeStat,
    /// An aggregate function folded an array into its partial.
    Accumulate,
    /// One execution step of an encoding, i.e. a decode.
    ExecuteEncoding,
}

/// One recorded event.
#[derive(Clone, Debug)]
pub struct StatsProbeEvent {
    /// What was done.
    pub kind: StatsProbeKind,
    /// The statistic or aggregate function; empty for an execution.
    pub what: String,
    /// The encoding of the array it was done to.
    pub encoding: String,
    /// The array's dtype.
    pub dtype: String,
    /// The array's length.
    pub len: usize,
}

/// Clear the recorded events and start recording.
pub fn start() {
    EVENTS.lock().clear();
    ACTIVE.store(true, Ordering::SeqCst);
}

/// Stop recording and return everything recorded since [`start`].
pub fn stop() -> Vec<StatsProbeEvent> {
    ACTIVE.store(false, Ordering::SeqCst);
    std::mem::take(&mut *EVENTS.lock())
}

#[inline]
pub(crate) fn record(kind: StatsProbeKind, what: impl Display, array: &ArrayRef) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let event = StatsProbeEvent {
        kind,
        what: what.to_string(),
        encoding: array.encoding_id().to_string(),
        dtype: array.dtype().to_string(),
        len: array.len(),
    };
    EVENTS.lock().push(event);
}
