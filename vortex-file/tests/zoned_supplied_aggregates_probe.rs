// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A zoned write with supplied partials computes no statistic, accumulates no aggregate and
//! decodes nothing. The stats probe is process-wide, so this binary holds a single test.

#![cfg(feature = "zstd")]
#![expect(clippy::tests_outside_test_module)]
#![allow(clippy::use_debug, reason = "the test prints what the probe saw")]

mod common;
mod supplied;

use std::collections::BTreeMap;

use vortex_array::test_harness::stats_probe;
use vortex_array::test_harness::stats_probe::StatsProbeEvent;
use vortex_array::test_harness::stats_probe::StatsProbeKind;
use vortex_error::VortexResult;

use crate::supplied::*;

/// Events on arrays this short are zone-table rows or partial scalars, not column data.
const DATA_SCALE: usize = 16;

fn data_scale(events: &[StatsProbeEvent]) -> Vec<&StatsProbeEvent> {
    events
        .iter()
        .filter(|event| event.len > DATA_SCALE)
        .collect()
}

fn summarize(label: &str, events: &[StatsProbeEvent]) {
    let mut groups = BTreeMap::<(String, String, String, String), (usize, usize)>::new();
    for event in events {
        let group = groups
            .entry((
                format!("{:?}", event.kind),
                event.what.clone(),
                event.encoding.clone(),
                event.dtype.clone(),
            ))
            .or_default();
        group.0 += 1;
        group.1 += event.len;
    }
    eprintln!(
        "[zy615] probe {label}: {} events, {} on data-scale arrays",
        events.len(),
        data_scale(events).len()
    );
    for ((kind, what, encoding, dtype), (count, rows)) in groups {
        eprintln!(
            "[zy615] probe {label}:   {kind} {what:?} on {encoding} {dtype}: {count} events, {rows} rows"
        );
    }
}

fn count(events: &[&StatsProbeEvent], kind: StatsProbeKind) -> usize {
    events.iter().filter(|event| event.kind == kind).count()
}

#[tokio::test]
async fn the_supplied_path_computes_and_decodes_nothing() -> VortexResult<()> {
    // Control: the computed path must show up, or a silent probe proves nothing.
    stats_probe::start();
    write_computed().await?;
    let computed = stats_probe::stop();
    summarize("computed", &computed);
    let computed_data = data_scale(&computed);
    assert!(
        count(&computed_data, StatsProbeKind::ComputeStat) > 0,
        "the probe saw no statistic computed"
    );
    assert!(
        count(&computed_data, StatsProbeKind::Accumulate) > 0,
        "the probe saw no accumulation"
    );

    // The supplied serial write.
    let prepared = prepare_zones(0..ZONES)?;
    let supplied_handles = new_supplied();
    stats_probe::start();
    let supplied_result = write_supplied(prepared, supplied_handles).await;
    let supplied = stats_probe::stop();
    supplied_result?;
    summarize("supplied", &supplied);

    // The supplied ranges and their stitch.
    let prepared = prepare_zones(0..ZONES)?;
    stats_probe::start();
    let stitched_result = write_supplied_stitched(prepared, &[0, 2, ZONES]).await;
    let stitched = stats_probe::stop();
    stitched_result?;
    summarize("supplied stitched", &stitched);

    // Today's writer handed the same compressed zones computes the partials from them: without
    // the uncompressed chunk's statistics, and with them inherited onto the compressed chunk.
    let prepared = prepare_zones_with(0..ZONES, false)?;
    stats_probe::start();
    let bare_result = write_uncompressing(prepared).await;
    let bare = stats_probe::stop();
    bare_result?;
    summarize("compressed chunks, writer computes", &bare);

    let prepared = prepare_zones_with(0..ZONES, true)?;
    stats_probe::start();
    let inherited_result = write_uncompressing(prepared).await;
    let inherited = stats_probe::stop();
    inherited_result?;
    summarize(
        "compressed chunks with inherited stats, writer computes",
        &inherited,
    );

    let supplied_data = data_scale(&supplied);
    let stitched_data = data_scale(&stitched);
    assert!(
        supplied_data.is_empty(),
        "the supplied write touched column data: {supplied_data:?}"
    );
    assert!(
        stitched_data.is_empty(),
        "the stitched supplied write touched column data: {stitched_data:?}"
    );
    Ok(())
}
