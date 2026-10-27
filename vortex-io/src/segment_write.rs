// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sinks that take a file as whole owned segments.

use std::io;
use std::sync::LazyLock;
use std::task::Context;
use std::task::Poll;

use vortex_buffer::ByteBuffer;
pub use vortex_buffer::segment::Piece;
pub use vortex_buffer::segment::Pieces;
pub use vortex_buffer::segment::Segment;
pub use vortex_buffer::segment::SegmentParts;
pub use vortex_buffer::segment::zero_page;

use crate::VortexWrite;

/// A sink that takes a file as whole segments and hands them back when it is done with their
/// bytes.
///
/// The file magic, each data segment and the footer each arrive in their own
/// [`write_segment`](SegmentWrite::write_segment), in file order. A sink can therefore record
/// exact boundaries at the start of each body (after [`Segment::leading_zeros`]), at each
/// segment's end, at the footer start and at the end of the file.
///
/// The trait has no `Send` bound, so a sink owned by one worker can be `!Send`. A writer driving
/// it is `Send` exactly when the sink is.
pub trait SegmentWrite {
    /// Ready when the sink accepts one more segment. `Pending` registers the waker, which is how
    /// the writer suspends until the sink has credit again.
    fn poll_credit(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>>;

    /// Hand over one segment. Only valid after [`poll_credit`](SegmentWrite::poll_credit)
    /// returned `Ready(Ok(()))`.
    fn write_segment(&mut self, segment: Segment) -> io::Result<()>;

    /// A segment whose bytes the sink no longer needs, for the writer to reuse.
    fn take_returned(&mut self) -> Option<Segment>;

    /// Ready when every segment handed over so far is durable as far as the sink can tell.
    fn poll_flush(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>>;
}

impl<S: SegmentWrite + ?Sized> SegmentWrite for &mut S {
    fn poll_credit(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        (**self).poll_credit(cx)
    }

    fn write_segment(&mut self, segment: Segment) -> io::Result<()> {
        (**self).write_segment(segment)
    }

    fn take_returned(&mut self) -> Option<Segment> {
        (**self).take_returned()
    }

    fn poll_flush(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        (**self).poll_flush(cx)
    }
}

/// Write `segment` to `write` with one [`VortexWrite::write_all`] per non-empty piece, then hand
/// back the segment emptied for reuse.
///
/// Every piece is written as a [`ByteBuffer`], as the writer did before segments existed, so
/// sinks that take the buffers themselves keep working. Nothing is copied: zeros are slices of
/// one shared zero buffer, data buffers are moved into the writes, and the framing storage
/// becomes one buffer that each framing piece slices. The framing storage is therefore not
/// reused on this path.
pub async fn write_segment_to<W: VortexWrite>(
    write: &mut W,
    segment: Segment,
) -> io::Result<Segment> {
    let SegmentParts {
        leading_zeros,
        pieces,
        mut buffers,
        framing,
    } = segment.into_parts();

    write_zeros(write, leading_zeros).await?;
    let framing = ByteBuffer::from(framing);
    let mut data = buffers.drain(..);
    for piece in &pieces {
        match *piece {
            Piece::Zeros(len) => write_zeros(write, len).await?,
            Piece::Buffer => {
                let buffer = data.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "a buffer piece has no buffer")
                })?;
                if !buffer.is_empty() {
                    write.write_all(buffer).await?;
                }
            }
            Piece::Framing { start, end } => {
                if start < end {
                    write
                        .write_all(framing.slice(start as usize..end as usize))
                        .await?;
                }
            }
        }
    }
    drop(data);

    Ok(Segment::reuse(SegmentParts {
        leading_zeros: 0,
        pieces,
        buffers,
        framing: Vec::new(),
    }))
}

/// One shared buffer over the zero page, so padding is written as slices of it.
fn zero_buffer() -> &'static ByteBuffer {
    static ZEROS: LazyLock<ByteBuffer> = LazyLock::new(|| ByteBuffer::copy_from(zero_page()));
    &ZEROS
}

async fn write_zeros<W: VortexWrite>(write: &mut W, len: u32) -> io::Result<()> {
    let zeros = zero_buffer();
    let mut remaining = len as usize;
    while remaining > 0 {
        let chunk = remaining.min(zeros.len());
        write.write_all(zeros.slice(0..chunk)).await?;
        remaining -= chunk;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn write_segment_to_writes_the_pieces_in_order() -> io::Result<()> {
        let mut segment = Segment::new();
        segment.set_leading_zeros(2);
        segment.push_buffer(ByteBuffer::copy_from(b"ab"));
        segment.push_zeros(1);
        segment.push_framing_bytes(b"fb");

        let mut out = Vec::new();
        let returned = write_segment_to(&mut out, segment).await?;
        assert_eq!(out, b"\0\0ab\0fb");
        assert!(returned.is_empty());
        assert!(returned.buffers().is_empty());
        Ok(())
    }
}
