// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use flatbuffers::Follow;
use flatbuffers::Verifiable;
use flatbuffers::VerifierOptions;
use flatbuffers::root_with_opts;
use vortex_buffer::ByteBuffer;
use vortex_buffer::segment::Piece;
use vortex_buffer::segment::Segment;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_flatbuffers::FILE_METADATA_VERIFIER_OPTIONS;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::FlatBufferRoot;
use vortex_flatbuffers::WriteFlatBuffer;
use vortex_flatbuffers::WriteFlatBufferExt;
use vortex_flatbuffers::dtype as fbd;
use vortex_flatbuffers::footer as fb;
use vortex_flatbuffers::layout as fbl;
use vortex_layout::LayoutContext;
use vortex_session::registry::ReadContext;
use vortex_utils::aliases::hash_map::HashMap;

use crate::EOF_SIZE;
use crate::Footer;
use crate::MAGIC_BYTES;
use crate::MAX_POSTSCRIPT_SIZE;
use crate::VERSION;
use crate::footer::SegmentSpec;
use crate::footer::file_layout::FooterFlatBufferWriter;
use crate::footer::postscript::Postscript;
use crate::footer::postscript::PostscriptMetadata;
use crate::footer::postscript::PostscriptSegment;

/// Serializes a [`Footer`] into footer buffers and the trailing postscript/EOF marker.
pub struct FooterSerializer {
    footer: Footer,
    metadata: HashMap<String, ByteBuffer>,
    exclude_dtype: bool,
    offset: u64,
    layout_ctx: LayoutContext,
    verifier_options: VerifierOptions,
}

impl FooterSerializer {
    pub(super) fn new(footer: Footer) -> Self {
        Self {
            footer,
            metadata: HashMap::default(),
            exclude_dtype: false,
            offset: 0,
            layout_ctx: LayoutContext::default(),
            verifier_options: FILE_METADATA_VERIFIER_OPTIONS,
        }
    }

    /// Verify serialized metadata flatbuffers with `options` instead of
    /// [`FILE_METADATA_VERIFIER_OPTIONS`], so tests can exercise the limits with small files.
    #[cfg(test)]
    pub(crate) fn with_verifier_options(mut self, options: VerifierOptions) -> Self {
        self.verifier_options = options;
        self
    }

    /// Serialize layouts through `layout_ctx`.
    ///
    /// Every layout written to the file is interned here, so a context restricted with
    /// [`LayoutContext::with_allowed_ids`] fails the write rather than emitting a layout
    /// outside the permitted set. The default context permits any layout.
    pub fn with_layout_context(mut self, layout_ctx: LayoutContext) -> Self {
        self.layout_ctx = layout_ctx;
        self
    }

    /// Update the offset used to generate absolute segment locations.
    ///
    /// This represents the byte position that the first buffer emitted by this serializer will be
    /// written to.
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    /// Exclude the DType from the serialized footer.
    /// If excluded, the reader must be provided the DType from an external source.
    pub fn exclude_dtype(mut self) -> Self {
        self.exclude_dtype = true;
        self
    }

    /// Whether to exclude the DType from the serialized footer.
    /// If excluded, the reader must be provided the DType from an external source.
    pub fn with_exclude_dtype(mut self, exclude_dtype: bool) -> Self {
        self.exclude_dtype = exclude_dtype;
        self
    }

    pub(crate) fn with_metadata_segments(mut self, metadata: HashMap<String, ByteBuffer>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Serialize the footer into a byte buffer that can later be deserialized as a [`Footer`].
    /// This can be helpful for storing some footer data out-of-band to accelerate opening a file.
    pub fn serialize(self) -> VortexResult<Vec<ByteBuffer>> {
        let (segment, _metadata, _approx_byte_size) = self.serialize_with_metadata()?;
        let mut data = segment.buffers().iter();
        Ok(segment
            .piece_list()
            .iter()
            .map(|piece| match *piece {
                Piece::Zeros(len) => ByteBuffer::zeroed(len as usize),
                Piece::Buffer => data
                    .next()
                    .vortex_expect("a buffer piece has a buffer")
                    .clone(),
                Piece::Framing { start, end } => {
                    ByteBuffer::copy_from(&segment.framing()[start as usize..end as usize])
                }
            })
            .collect())
    }

    /// Serialize the footer as one segment, starting at the configured offset.
    pub(crate) fn serialize_with_metadata(
        mut self,
    ) -> VortexResult<(Segment, super::MetadataSegments, usize)> {
        #[cfg(feature = "bench-tail")]
        let mut stamps = SerializerStamps::default();
        #[cfg(feature = "bench-tail")]
        let mut lap = bench_tail::Lap::start();
        let mut buffers = Segment::new();

        let (metadata_segments, metadata_locators) = if self.metadata.is_empty() {
            let locators = self
                .footer
                .metadata_segments()
                .map(|(key, segment)| (key.to_string(), *segment))
                .collect::<Vec<_>>();
            let segments = locators
                .iter()
                .map(|(key, segment)| PostscriptMetadata {
                    key: key.clone(),
                    segment: PostscriptSegment::from(segment),
                })
                .collect();
            (segments, locators)
        } else {
            let mut metadata = self.metadata.into_iter().collect::<Vec<_>>();
            // Metadata is stored in a HashMap, so sort keys before assigning segment offsets.
            metadata.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));

            let mut segments = Vec::with_capacity(metadata.len());
            let mut locators = Vec::with_capacity(metadata.len());
            for (key, metadata) in metadata {
                let segment = write_buffer(&mut self.offset, &mut buffers, metadata)?;
                locators.push((key.clone(), SegmentSpec::from(&segment)));
                segments.push(PostscriptMetadata { key, segment });
            }
            (segments, locators)
        };
        let metadata_storage_bytes = buffers.len();
        #[cfg(feature = "bench-tail")]
        {
            stamps.metadata_us = lap.lap();
        }

        let dtype_segment = if self.exclude_dtype {
            None
        } else {
            let (buffer, dtype_segment) = write_flatbuffer(&mut self.offset, self.footer.dtype())?;
            verify_readable::<fbd::DType>(&self.verifier_options, "dtype", &buffer)?;
            buffers.push_buffer(buffer);
            Some(dtype_segment)
        };
        #[cfg(feature = "bench-tail")]
        {
            stamps.dtype_us = lap.lap();
        }

        // TODO(ngates): we should separate the read/write side of Context since the write side
        //  doesn't need to look anything up in the registry.
        let layout_ctx = self.layout_ctx.clone();

        let (buffer, layout_segment) = write_flatbuffer(
            &mut self.offset,
            &self.footer.layout().flatbuffer_writer(&layout_ctx),
        )?;
        #[cfg(feature = "bench-tail")]
        {
            stamps.layout_write_us = lap.lap();
            stamps.layout_bytes = u64::try_from(buffer.len()).unwrap_or(u64::MAX);
        }
        verify_readable::<fbl::Layout>(&self.verifier_options, "layout", &buffer)?;
        #[cfg(feature = "bench-tail")]
        {
            stamps.layout_verify_us = lap.lap();
        }
        buffers.push_buffer(buffer);

        let statistics_segment = match self.footer.statistics() {
            None => None,
            Some(stats) if stats.stats_sets().is_empty() => None,
            Some(stats) => {
                let (buffer, stats_segment) = write_flatbuffer(&mut self.offset, stats)?;
                verify_readable::<fb::FileStatistics>(
                    &self.verifier_options,
                    "file statistics",
                    &buffer,
                )?;
                buffers.push_buffer(buffer);
                Some(stats_segment)
            }
        };
        #[cfg(feature = "bench-tail")]
        {
            stamps.statistics_us = lap.lap();
        }

        let (buffer, footer_segment) = write_flatbuffer(
            &mut self.offset,
            &FooterFlatBufferWriter {
                ctx: self.footer.array_read_ctx.clone(),
                layout_ctx: ReadContext::new(layout_ctx.to_ids()),
                segment_specs: Arc::clone(&self.footer.segments),
            },
        )?;
        #[cfg(feature = "bench-tail")]
        {
            stamps.footer_write_us = lap.lap();
            stamps.footer_bytes = u64::try_from(buffer.len()).unwrap_or(u64::MAX);
        }
        verify_readable::<fb::Footer>(&self.verifier_options, "footer", &buffer)?;
        #[cfg(feature = "bench-tail")]
        {
            stamps.footer_verify_us = lap.lap();
        }
        buffers.push_buffer(buffer);

        // Assemble the postscript, and write it manually to avoid any framing.
        let postscript = Postscript {
            dtype: dtype_segment,
            layout: layout_segment,
            statistics: statistics_segment,
            footer: footer_segment,
            metadata: metadata_segments,
        };
        let postscript_buffer = postscript.write_flatbuffer_bytes()?;
        verify_readable::<fb::Postscript>(
            &self.verifier_options,
            "postscript",
            &postscript_buffer,
        )?;
        if postscript_buffer.len() > MAX_POSTSCRIPT_SIZE as usize {
            Err(vortex_err!(
                "Postscript is too large ({} bytes); max postscript size is {}",
                postscript_buffer.len(),
                MAX_POSTSCRIPT_SIZE
            ))?;
        }

        let postscript_len = u16::try_from(postscript_buffer.len())
            .vortex_expect("Postscript already verified to fit into u16");
        buffers.push_buffer(postscript_buffer.into_inner());

        // And finally, the EOF 8-byte footer.
        let mut eof = [0u8; EOF_SIZE];
        eof[0..2].copy_from_slice(&VERSION.to_le_bytes());
        eof[2..4].copy_from_slice(&postscript_len.to_le_bytes());
        eof[4..8].copy_from_slice(&MAGIC_BYTES);
        buffers.push_framing_bytes(&eof);

        let approx_byte_size = usize::try_from(buffers.len() - metadata_storage_bytes)
            .vortex_expect("the footer fits in memory");
        #[cfg(feature = "bench-tail")]
        {
            stamps.postscript_us = lap.lap();
            bench_tail::set(stamps);
        }
        Ok((buffers, metadata_locators.into(), approx_byte_size))
    }
}

impl From<&PostscriptSegment> for SegmentSpec {
    fn from(value: &PostscriptSegment) -> Self {
        Self {
            offset: value.offset,
            length: value.length,
            alignment: value.alignment,
        }
    }
}

impl From<&SegmentSpec> for PostscriptSegment {
    fn from(value: &SegmentSpec) -> Self {
        Self {
            offset: value.offset,
            length: value.length,
            alignment: value.alignment,
        }
    }
}

/// Fails the write when readers would reject a serialized metadata flatbuffer.
///
/// Writers use [`FILE_METADATA_VERIFIER_OPTIONS`]. Readers verify each metadata flatbuffer with
/// options that are never stricter, so a buffer that passes here can be opened. The footer is
/// serialized after every data segment and before the file is finished, so a rejected buffer fails
/// the write before the file becomes readable. The returned error keeps the flatbuffers error
/// (for example `TooManyTables`) as its source.
fn verify_readable<'buf, T: Verifiable + Follow<'buf> + 'buf>(
    options: &VerifierOptions,
    name: &'static str,
    buffer: &'buf [u8],
) -> VortexResult<()> {
    root_with_opts::<T>(options, buffer)
        .map(drop)
        .map_err(|error| {
            VortexError::from(error).with_context(format!(
                "Vortex readers would reject the {name} flatbuffer ({} bytes; limits: {} tables, \
                 depth {}, apparent size {} bytes)",
                buffer.len(),
                options.max_tables,
                options.max_depth,
                options.max_apparent_size,
            ))
        })
}

fn write_flatbuffer<F: FlatBufferRoot + WriteFlatBuffer>(
    offset: &mut u64,
    flatbuffer: &F,
) -> VortexResult<(ByteBuffer, PostscriptSegment)> {
    let buffer = flatbuffer.write_flatbuffer_bytes()?;
    let length = u32::try_from(buffer.len())
        .map_err(|_| vortex_err!("flatbuffer length exceeds maximum u32"))?;

    let segment = PostscriptSegment {
        offset: *offset,
        length,
        alignment: FlatBuffer::alignment(),
    };

    *offset += u64::from(length);

    Ok((buffer.into_inner(), segment))
}

fn write_buffer(
    offset: &mut u64,
    out: &mut Segment,
    buffer: ByteBuffer,
) -> VortexResult<PostscriptSegment> {
    let length = u32::try_from(buffer.len())
        .map_err(|_| vortex_err!("metadata segment length exceeds maximum u32"))?;
    let alignment = buffer.alignment();

    let padding = offset.next_multiple_of(alignment.as_usize() as u64) - *offset;
    let segment_offset = *offset + padding;

    let segment = PostscriptSegment {
        offset: segment_offset,
        length,
        alignment,
    };

    *offset += padding + u64::from(length);

    out.push_zeros(
        u32::try_from(padding).map_err(|_| vortex_err!("metadata padding exceeds maximum u32"))?,
    );
    out.push_buffer(buffer);

    Ok(segment)
}

/// ZY-184 bet 3 (Lease 6): test-only stamps of where [`FooterSerializer::serialize_with_metadata`]
/// spent its time, in microseconds, and the sizes of the two large flatbuffers.
#[cfg(feature = "bench-tail")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SerializerStamps {
    /// The metadata segments.
    pub metadata_us: u64,
    /// The dtype flatbuffer, written and verified.
    pub dtype_us: u64,
    /// Writing the layout flatbuffer.
    pub layout_write_us: u64,
    /// The layout flatbuffer's bytes.
    pub layout_bytes: u64,
    /// Verifying the layout flatbuffer as a reader would.
    pub layout_verify_us: u64,
    /// The file statistics flatbuffer, if any.
    pub statistics_us: u64,
    /// Writing the footer flatbuffer (the segment specs).
    pub footer_write_us: u64,
    /// The footer flatbuffer's bytes.
    pub footer_bytes: u64,
    /// Verifying the footer flatbuffer as a reader would.
    pub footer_verify_us: u64,
    /// The postscript, its verify, and the EOF.
    pub postscript_us: u64,
}

/// ZY-184 bet 3 (Lease 6): the last serializer's stamps on this thread, for the stitched writer
/// to pick up right after it serialized.
#[cfg(feature = "bench-tail")]
pub(crate) mod bench_tail {
    use std::cell::Cell;
    use std::time::Instant;

    use super::SerializerStamps;

    thread_local! {
        static LAST: Cell<SerializerStamps> = const { Cell::new(SerializerStamps {
            metadata_us: 0,
            dtype_us: 0,
            layout_write_us: 0,
            layout_bytes: 0,
            layout_verify_us: 0,
            statistics_us: 0,
            footer_write_us: 0,
            footer_bytes: 0,
            footer_verify_us: 0,
            postscript_us: 0,
        }) };
    }

    /// Records the stamps of the serializer that just finished on this thread.
    pub(crate) fn set(stamps: SerializerStamps) {
        LAST.with(|cell| cell.set(stamps));
    }

    /// The stamps of the last serializer on this thread, leaving the default behind.
    pub(crate) fn take() -> SerializerStamps {
        LAST.with(Cell::take)
    }

    /// A stopwatch that hands out the microseconds since its start or its previous lap.
    pub(crate) struct Lap(Instant);

    impl Lap {
        pub(crate) fn start() -> Self {
            Self(Instant::now())
        }

        /// Microseconds since the previous lap (or the start); the next lap starts now.
        pub(crate) fn lap(&mut self) -> u64 {
            let now = Instant::now();
            let us = u64::try_from(now.duration_since(self.0).as_micros()).unwrap_or(u64::MAX);
            self.0 = now;
            us
        }
    }
}
