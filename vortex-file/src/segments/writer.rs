// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::SinkExt;
use futures::channel::mpsc::Sender;
use futures::lock::Mutex as AsyncMutex;
use parking_lot::Mutex;
use vortex_buffer::segment::Segment;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSink;
use vortex_layout::sequence::SequenceId;

use crate::footer::SegmentSpec;

/// Segments kept for reuse. A file writer has at most a few segments in flight, so a handful
/// covers steady state without holding much memory.
const MAX_RECYCLED_SEGMENTS: usize = 8;

pub struct BufferedSegmentSink {
    segments: AsyncMutex<Sender<Segment>>,
    byte_offset: AtomicU64,
    segment_specs: Mutex<Vec<SegmentSpec>>,
    recycled: SegmentRecycler,
}

/// Segments the output is done with, kept for the sink to hand out again.
///
/// It is separate from [`BufferedSegmentSink`] because the sink owns the channel's sender: the
/// output side must not keep the sink alive, or the channel never closes.
#[derive(Clone, Default)]
pub struct SegmentRecycler(Arc<Mutex<Vec<Segment>>>);

impl SegmentRecycler {
    /// Keep a segment the output is done with, so a later [`SegmentSink::new_segment`] reuses
    /// its storage.
    pub fn recycle(&self, mut segment: Segment) {
        let mut recycled = self.0.lock();
        if recycled.len() < MAX_RECYCLED_SEGMENTS {
            segment.clear();
            recycled.push(segment);
        }
    }

    fn take(&self) -> Segment {
        self.0.lock().pop().unwrap_or_default()
    }
}

impl BufferedSegmentSink {
    pub fn new(send: Sender<Segment>, byte_offset: u64) -> Self {
        Self {
            segments: AsyncMutex::new(send),
            byte_offset: AtomicU64::new(byte_offset),
            segment_specs: Default::default(),
            recycled: Default::default(),
        }
    }

    /// Close the sink, returning the segment specs and the final byte offset.
    pub fn segment_specs(&self) -> Arc<[SegmentSpec]> {
        let specs = self.segment_specs.lock();
        specs.clone().into()
    }

    /// The pool that [`SegmentSink::new_segment`] takes segments from.
    pub fn recycler(&self) -> SegmentRecycler {
        self.recycled.clone()
    }
}

#[async_trait]
impl SegmentSink for BufferedSegmentSink {
    fn new_segment(&self) -> Segment {
        self.recycled.take()
    }

    async fn write(
        &self,
        mut sequence_id: SequenceId,
        mut segment: Segment,
    ) -> VortexResult<SegmentId> {
        // We wait for all segment IDs before this one to be dropped. Then while we hold a strong
        // reference to this one, we essentially have an exclusive lock on the segment writer.
        sequence_id.collapse().await;

        let segment_id = {
            let mut specs = self.segment_specs.lock();
            let segment_id = SegmentId::from(
                u32::try_from(specs.len())
                    .map_err(|_| vortex_err!("Too mant segments, u32 overflow"))?,
            );

            // The segment is written contiguously, so only the alignment of its body can be
            // respected. The serializer pads the buffers inside the body.
            let alignment = segment.alignment();
            let length = u32::try_from(segment.len())
                .map_err(|_| vortex_err!("segment buffer length exceeds maximum u32"))?;

            // Add any padding required to align the segment.
            let byte_offset = self.byte_offset.load(Ordering::Relaxed);
            let padding = byte_offset.next_multiple_of(alignment.as_usize() as u64) - byte_offset;
            let offset = byte_offset + padding;
            specs.push(SegmentSpec {
                offset,
                length,
                alignment,
            });

            self.byte_offset
                .store(byte_offset + padding + u64::from(length), Ordering::Relaxed);

            segment.set_leading_zeros(
                u32::try_from(padding)
                    .map_err(|_| vortex_err!("segment padding exceeds maximum u32"))?,
            );
            segment_id
        };

        // Collapse serializes writers, so this guard is uncontended. Borrow the same sender
        // across the bounded send instead of allocating sender state for each segment. The
        // guard also retains the sender if this write is cancelled while awaiting the output.
        drop(self.segments.lock().await.send(segment).await);

        Ok(segment_id)
    }
}
