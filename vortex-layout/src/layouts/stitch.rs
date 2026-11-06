// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Join the layout trees of consecutive range writes into the tree one write would have built.
//!
//! A range write runs the file's strategy over a whole number of zones with its zone maps
//! deferred (see [`LayoutWriterContext::with_zone_deferral`]) and numbers its segments from 0.
//! The stitch renumbers each range's segments by the ids written before it, concatenates the
//! chunk children of each column, and builds every zone table from the ranges' partials.
//!
//! Only the layouts the chunked, struct, zoned and flat strategies build are joined; anything
//! else fails rather than producing a different tree.

use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutWriterContext;
use crate::layout_children;
use crate::layouts::chunked::Chunked;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::Flat;
use crate::layouts::flat::FlatLayout;
use crate::layouts::struct_::Struct;
use crate::layouts::struct_::StructLayout;
use crate::layouts::zoned::writer::DeferredZones;
use crate::layouts::zoned::writer::write_deferred_zones;
use crate::segments::SegmentId;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequencePointer;

/// One range write's layout tree, its deferred zone maps, and the id of its first segment in
/// the stitched file.
pub struct RangeLayout {
    /// The range's layout tree, with segment ids numbered from 0.
    pub layout: LayoutRef,
    /// The zone maps the range deferred.
    pub zones: Vec<DeferredZones>,
    /// The stitched id of the range's segment 0.
    pub first_segment_id: u32,
}

/// Join consecutive range layouts into one tree.
///
/// Zone tables are written to `segment_sink` in the order a single write emits them: depth
/// first, in field order, after every data segment.
pub async fn stitch_range_layouts(
    ranges: Vec<RangeLayout>,
    ctx: LayoutWriterContext,
    segment_sink: SegmentSinkRef,
    mut eof: SequencePointer,
    session: &VortexSession,
) -> VortexResult<LayoutRef> {
    vortex_ensure!(!ranges.is_empty(), "a stitch needs at least one range");
    #[cfg(feature = "bench-tail")]
    bench_tail::reset_deferred_zones();
    let bases: Vec<u32> = ranges.iter().map(|range| range.first_segment_id).collect();
    let mut zones: Vec<Vec<Option<DeferredZones>>> = Vec::with_capacity(ranges.len());
    let mut roots = Vec::with_capacity(ranges.len());
    for range in ranges {
        roots.push(range.layout);
        zones.push(range.zones.into_iter().map(Some).collect());
    }
    let mut state = Stitch {
        bases,
        zones,
        ctx,
        segment_sink,
        session,
    };
    let layout = state.join(roots, &mut eof, true).await?;
    for (index, range) in state.zones.iter().enumerate() {
        vortex_ensure!(
            range.iter().all(Option::is_none),
            "range {index} deferred a zone map that is not in its layout tree"
        );
    }
    Ok(layout)
}

struct Stitch<'a> {
    bases: Vec<u32>,
    zones: Vec<Vec<Option<DeferredZones>>>,
    ctx: LayoutWriterContext,
    segment_sink: SegmentSinkRef,
    session: &'a VortexSession,
}

impl<'a> Stitch<'a> {
    /// Join the nodes at one position of every range's tree.
    fn join<'s>(
        &'s mut self,
        nodes: Vec<LayoutRef>,
        eof: &'s mut SequencePointer,
        check_zones: bool,
    ) -> BoxFuture<'s, VortexResult<LayoutRef>>
    where
        'a: 's,
    {
        async move {
            let first = nodes
                .first()
                .ok_or_else(|| vortex_err!("a stitch position has no ranges"))?;
            let dtype = first.dtype().clone();
            for node in &nodes {
                vortex_ensure!(
                    node.dtype() == &dtype,
                    "range layouts disagree on dtype: {} and {}",
                    node.dtype(),
                    dtype
                );
            }

            if check_zones {
                let deferred = self.take_zones(&nodes)?;
                if let Some(parts) = deferred {
                    let data = self.join(nodes, eof, false).await?;
                    #[cfg(feature = "bench-tail")]
                    let _deferred_zones_timer = bench_tail::DeferredZonesTimer::start();
                    return write_deferred_zones(
                        parts,
                        data,
                        self.ctx.clone(),
                        Arc::clone(&self.segment_sink),
                        eof.split_off(),
                        self.session,
                    )
                    .await;
                }
            }

            if first.is::<Struct>() {
                let nfields = first.nchildren();
                let mut columns: Vec<Vec<LayoutRef>> = (0..nfields)
                    .map(|_| Vec::with_capacity(nodes.len()))
                    .collect();
                let mut rows = 0_u64;
                for node in &nodes {
                    vortex_ensure!(
                        node.is::<Struct>() && node.nchildren() == nfields,
                        "range layouts disagree on struct shape"
                    );
                    rows = rows
                        .checked_add(node.row_count())
                        .ok_or_else(|| vortex_err!("stitched row count overflow"))?;
                    for (column, child) in columns.iter_mut().zip(node.children()?) {
                        column.push(child);
                    }
                }
                let mut children = Vec::with_capacity(nfields);
                for column in columns {
                    children.push(self.join(column, eof, true).await?);
                }
                return Ok(StructLayout::new(rows, dtype, children).into_layout());
            }

            self.concat(nodes)
        }
        .boxed()
    }

    /// Take each range's deferred zone map for this position, if the position is zoned.
    fn take_zones(&mut self, nodes: &[LayoutRef]) -> VortexResult<Option<Vec<DeferredZones>>> {
        let mut found = Vec::with_capacity(nodes.len());
        for (node, zones) in nodes.iter().zip(self.zones.iter_mut()) {
            let slot = zones.iter_mut().find(|slot| {
                slot.as_ref()
                    .is_some_and(|zones| Arc::ptr_eq(zones.data(), node))
            });
            found.push(slot.and_then(Option::take));
        }
        if found.iter().all(Option::is_none) {
            return Ok(None);
        }
        if found.iter().any(Option::is_none) {
            vortex_bail!("ranges disagree on whether a column is zoned");
        }
        Ok(Some(found.into_iter().flatten().collect()))
    }

    /// Concatenate the flat chunks of each range, renumbering their segments.
    fn concat(&self, nodes: Vec<LayoutRef>) -> VortexResult<LayoutRef> {
        let dtype = nodes[0].dtype().clone();
        let mut children = Vec::new();
        let mut rows = 0_u64;
        for (node, base) in nodes.iter().zip(self.bases.iter().copied()) {
            if node.is::<Flat>() {
                children.push(renumber(node, base)?);
            } else if node.is::<Chunked>() {
                for child in node.children()? {
                    vortex_ensure!(
                        child.is::<Flat>(),
                        "stitch joins only chunked layouts of flat chunks, found {}",
                        child.encoding_id()
                    );
                    children.push(renumber(&child, base)?);
                }
            } else {
                vortex_bail!("stitch cannot join layout {}", node.encoding_id());
            }
            rows = rows
                .checked_add(node.row_count())
                .ok_or_else(|| vortex_err!("stitched row count overflow"))?;
        }
        if children.len() == 1 {
            return children
                .pop()
                .ok_or_else(|| vortex_err!("stitched chunk list lost its only child"));
        }
        Ok(ChunkedLayout::new(rows, dtype, layout_children(children)).into_layout())
    }
}

fn renumber(node: &LayoutRef, base: u32) -> VortexResult<LayoutRef> {
    let flat = node.as_::<Flat>();
    let id = flat
        .segment_id()
        .checked_add(base)
        .ok_or_else(|| vortex_err!("stitched segment id overflow"))?;
    Ok(FlatLayout::new_with_metadata(
        flat.row_count(),
        flat.dtype().clone(),
        SegmentId::from(id),
        flat.array_ctx().clone(),
        flat.array_tree().cloned(),
    )
    .into_layout())
}

/// ZY-184 bet 3 (Lease 6): test-only stamps of the time a stitch spends in
/// [`write_deferred_zones`], kept per thread, reset by [`stitch_range_layouts`] and read by the
/// file writer right after it on the same thread.
#[cfg(feature = "bench-tail")]
pub mod bench_tail {
    use std::cell::Cell;
    use std::time::Instant;

    thread_local! {
        static DEFERRED_ZONES: Cell<(u64, u32)> = const { Cell::new((0, 0)) };
    }

    /// Forgets this thread's deferred-zone total; the next stitch starts from zero.
    pub fn reset_deferred_zones() {
        DEFERRED_ZONES.with(|cell| cell.set((0, 0)));
    }

    /// Microseconds spent in `write_deferred_zones` on this thread since the last reset, and
    /// the number of calls.
    pub fn deferred_zones() -> (u64, u32) {
        DEFERRED_ZONES.with(Cell::get)
    }

    /// Adds the time since its start to this thread's deferred-zone total when dropped.
    pub(super) struct DeferredZonesTimer(Instant);

    impl DeferredZonesTimer {
        pub(super) fn start() -> Self {
            Self(Instant::now())
        }
    }

    impl Drop for DeferredZonesTimer {
        fn drop(&mut self) {
            let us = u64::try_from(self.0.elapsed().as_micros()).unwrap_or(u64::MAX);
            DEFERRED_ZONES.with(|cell| {
                let (total, calls) = cell.get();
                cell.set((total.saturating_add(us), calls.saturating_add(1)));
            });
        }
    }
}
