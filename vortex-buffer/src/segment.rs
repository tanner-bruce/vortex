// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One segment of a file, owned, with its padding described by length.
//!
//! A serialized array is a run of data buffers, zero padding, and a flatbuffer with its length.
//! A [`Segment`] holds the data buffers as moved [`ByteBuffer`]s, the flatbuffer and length in one
//! owned `Vec<u8>`, and padding as a count of zero bytes. A sink reads it in file order through
//! [`Segment::pieces`], or takes it apart with [`Segment::into_parts`].

use std::mem;
use std::ops::Range;

use vortex_error::VortexExpect;

use crate::Alignment;
use crate::ByteBuffer;

/// Length of the zero page, and so of the longest zero slice [`Segment::pieces`] yields.
const ZERO_PAGE_PIECE: u32 = 64 * 1024;
const ZERO_PAGE_LEN: usize = ZERO_PAGE_PIECE as usize;

static ZERO_PAGE: [u8; ZERO_PAGE_LEN] = [0; ZERO_PAGE_LEN];

/// A static page of zero bytes that padding pieces are read from.
pub fn zero_page() -> &'static [u8] {
    &ZERO_PAGE
}

/// One piece of a [`Segment`], in file order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Piece {
    /// This many zero bytes.
    Zeros(u32),
    /// The next of the segment's buffers.
    Buffer,
    /// The given range of the segment's framing bytes.
    Framing {
        /// First byte of the range.
        start: u32,
        /// One past the last byte of the range.
        end: u32,
    },
}

/// One segment as it lies in the file, owned, with its padding described by length.
///
/// The segment starts with [`Segment::leading_zeros`] zero bytes, which align its body in the
/// file, followed by its pieces. A sink that is done with the bytes can hand the segment back to
/// the writer, which [`clear`](Segment::clear)s it and reuses its storage.
#[derive(Debug)]
pub struct Segment {
    alignment: Alignment,
    leading_zeros: u32,
    len: u64,
    pieces: Vec<Piece>,
    buffers: Vec<ByteBuffer>,
    framing: Vec<u8>,
}

impl Default for Segment {
    fn default() -> Self {
        Self::new()
    }
}

/// The owned storage of a [`Segment`], for a sink that needs the buffers themselves.
#[derive(Debug)]
pub struct SegmentParts {
    /// Zero bytes before the first piece.
    pub leading_zeros: u32,
    /// The pieces in file order.
    pub pieces: Vec<Piece>,
    /// The buffers that [`Piece::Buffer`] refers to, in order.
    pub buffers: Vec<ByteBuffer>,
    /// The bytes that [`Piece::Framing`] ranges refer to.
    pub framing: Vec<u8>,
}

impl Segment {
    /// An empty segment with no alignment requirement.
    pub const fn new() -> Self {
        Self {
            alignment: Alignment::none(),
            leading_zeros: 0,
            len: 0,
            pieces: Vec::new(),
            buffers: Vec::new(),
            framing: Vec::new(),
        }
    }

    /// The alignment the segment's body needs in the file.
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Set the alignment the segment's body needs in the file.
    pub fn set_alignment(&mut self, alignment: Alignment) {
        self.alignment = alignment;
    }

    /// Zero bytes before the body, which a sink adds to align it.
    pub fn leading_zeros(&self) -> u32 {
        self.leading_zeros
    }

    /// Set the zero bytes before the body.
    pub fn set_leading_zeros(&mut self, leading_zeros: u32) {
        self.leading_zeros = leading_zeros;
    }

    /// The length of the body, without the leading zeros.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the body is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The length of the leading zeros and the body together.
    pub fn padded_len(&self) -> u64 {
        u64::from(self.leading_zeros) + self.len
    }

    /// Append `len` zero bytes. Nothing is appended for zero.
    pub fn push_zeros(&mut self, len: u32) {
        if len == 0 {
            return;
        }
        if let Some(Piece::Zeros(previous)) = self.pieces.last_mut()
            && let Some(merged) = previous.checked_add(len)
        {
            *previous = merged;
        } else {
            self.pieces.push(Piece::Zeros(len));
        }
        self.len += u64::from(len);
    }

    /// Append a data buffer, moved into the segment.
    pub fn push_buffer(&mut self, buffer: ByteBuffer) {
        self.len += buffer.len() as u64;
        self.pieces.push(Piece::Buffer);
        self.buffers.push(buffer);
    }

    /// Append a copy of `bytes` as framing.
    ///
    /// # Panics
    ///
    /// Panics if the framing grows past `u32::MAX` bytes.
    pub fn push_framing_bytes(&mut self, bytes: &[u8]) {
        let start = self.framing.len();
        self.framing.extend_from_slice(bytes);
        self.push_framing_range(start..self.framing.len());
    }

    /// Take the framing storage, empty, to build bytes in place. Hand it back with
    /// [`Segment::put_framing`].
    ///
    /// # Panics
    ///
    /// Panics if a framing piece already refers to the storage.
    pub fn take_framing(&mut self) -> Vec<u8> {
        assert!(
            !self
                .pieces
                .iter()
                .any(|piece| matches!(piece, Piece::Framing { .. })),
            "framing storage is taken before any framing piece refers to it"
        );
        let mut framing = mem::take(&mut self.framing);
        framing.clear();
        framing
    }

    /// Put back storage from [`Segment::take_framing`], appending `range` of it as framing.
    ///
    /// # Panics
    ///
    /// Panics if `range` is not within `framing`, or if `framing` is longer than `u32::MAX` bytes.
    pub fn put_framing(&mut self, framing: Vec<u8>, range: Range<usize>) {
        assert!(
            range.start <= range.end && range.end <= framing.len(),
            "framing range {range:?} is within {} bytes",
            framing.len()
        );
        self.framing = framing;
        self.push_framing_range(range);
    }

    fn push_framing_range(&mut self, range: Range<usize>) {
        if range.is_empty() {
            return;
        }
        let start = u32::try_from(range.start).vortex_expect("framing fits in u32");
        let end = u32::try_from(range.end).vortex_expect("framing fits in u32");
        self.len += u64::from(end - start);
        self.pieces.push(Piece::Framing { start, end });
    }

    /// The pieces in file order, without the leading zeros.
    pub fn piece_list(&self) -> &[Piece] {
        &self.pieces
    }

    /// The data buffers in file order.
    pub fn buffers(&self) -> &[ByteBuffer] {
        &self.buffers
    }

    /// The framing bytes that [`Piece::Framing`] ranges refer to.
    pub fn framing(&self) -> &[u8] {
        &self.framing
    }

    /// The bytes of the segment in file order, starting with the leading zeros. Empty slices are
    /// skipped, and long runs of zeros come as several slices of the zero page.
    pub fn pieces(&self) -> Pieces<'_> {
        Pieces {
            segment: self,
            zeros: self.leading_zeros,
            next_piece: 0,
            next_buffer: 0,
        }
    }

    /// The bytes of the body in file order, without the leading zeros.
    pub fn body_pieces(&self) -> Pieces<'_> {
        Pieces {
            segment: self,
            zeros: 0,
            next_piece: 0,
            next_buffer: 0,
        }
    }

    /// The number of data buffers whose storage another buffer still shares, so releasing the
    /// segment will not free them.
    pub fn shared_buffers(&self) -> usize {
        self.buffers
            .iter()
            .filter(|buffer| buffer.is_shared())
            .count()
    }

    /// Empty the segment for reuse, keeping its storage. The buffers are released.
    pub fn clear(&mut self) {
        self.alignment = Alignment::none();
        self.leading_zeros = 0;
        self.len = 0;
        self.pieces.clear();
        self.buffers.clear();
        self.framing.clear();
    }

    /// An empty segment that reuses the storage of `parts`. Their contents are released.
    pub fn reuse(parts: SegmentParts) -> Self {
        let SegmentParts {
            pieces,
            buffers,
            framing,
            ..
        } = parts;
        let mut segment = Self {
            alignment: Alignment::none(),
            leading_zeros: 0,
            len: 0,
            pieces,
            buffers,
            framing,
        };
        segment.clear();
        segment
    }

    /// Take the segment apart.
    pub fn into_parts(self) -> SegmentParts {
        SegmentParts {
            leading_zeros: self.leading_zeros,
            pieces: self.pieces,
            buffers: self.buffers,
            framing: self.framing,
        }
    }
}

/// The bytes of a [`Segment`] in file order; see [`Segment::pieces`].
#[derive(Debug)]
pub struct Pieces<'a> {
    segment: &'a Segment,
    zeros: u32,
    next_piece: usize,
    next_buffer: usize,
}

impl<'a> Iterator for Pieces<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        loop {
            if self.zeros > 0 {
                let len = self.zeros.min(ZERO_PAGE_PIECE);
                self.zeros -= len;
                return Some(&ZERO_PAGE[..len as usize]);
            }
            let piece = self.segment.pieces.get(self.next_piece)?;
            self.next_piece += 1;
            let bytes = match *piece {
                Piece::Zeros(len) => {
                    self.zeros = len;
                    continue;
                }
                Piece::Buffer => {
                    let buffer = &self.segment.buffers[self.next_buffer];
                    self.next_buffer += 1;
                    buffer.as_slice()
                }
                Piece::Framing { start, end } => {
                    &self.segment.framing[start as usize..end as usize]
                }
            };
            if !bytes.is_empty() {
                return Some(bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn concat(pieces: Pieces<'_>) -> Vec<u8> {
        pieces.flat_map(|bytes| bytes.iter().copied()).collect()
    }

    #[test]
    fn pieces_follow_file_order() {
        let mut segment = Segment::new();
        segment.push_buffer(ByteBuffer::copy_from(b"ab"));
        segment.push_zeros(2);
        segment.push_zeros(1);
        segment.push_buffer(ByteBuffer::empty());
        segment.push_framing_bytes(b"xyz");
        segment.set_leading_zeros(3);

        assert_eq!(segment.len(), 8);
        assert_eq!(segment.padded_len(), 11);
        assert_eq!(
            segment.piece_list(),
            &[
                Piece::Buffer,
                Piece::Zeros(3),
                Piece::Buffer,
                Piece::Framing { start: 0, end: 3 }
            ]
        );
        assert_eq!(concat(segment.pieces()), b"\0\0\0ab\0\0\0xyz");
        assert_eq!(concat(segment.body_pieces()), b"ab\0\0\0xyz");
        assert_eq!(segment.pieces().count(), 4, "the empty buffer is skipped");
    }

    #[test]
    fn long_zero_runs_come_from_the_zero_page() {
        let mut segment = Segment::new();
        segment.push_zeros(u32::try_from(ZERO_PAGE_LEN * 2 + 5).unwrap());
        let lens = segment.pieces().map(<[u8]>::len).collect::<Vec<_>>();
        assert_eq!(lens, [ZERO_PAGE_LEN, ZERO_PAGE_LEN, 5]);
    }

    #[test]
    fn framing_built_in_place_keeps_its_storage() {
        let mut segment = Segment::new();
        let mut framing = segment.take_framing();
        framing.extend_from_slice(b"..fb");
        segment.put_framing(framing, 2..4);
        assert_eq!(concat(segment.pieces()), b"fb");

        let capacity = segment.framing().len();
        segment.clear();
        assert!(segment.is_empty());
        assert!(segment.take_framing().capacity() >= capacity);
    }

    #[test]
    fn shared_buffers_counts_aliases() {
        let buffer = ByteBuffer::copy_from(b"abc");
        let mut segment = Segment::new();
        segment.push_buffer(buffer.clone());
        segment.push_buffer(ByteBuffer::copy_from(b"def"));
        assert_eq!(segment.shared_buffers(), 1);
        drop(buffer);
        assert_eq!(segment.shared_buffers(), 0);
    }
}
