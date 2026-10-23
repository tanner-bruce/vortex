// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metadata flatbuffer verification limits for large files.
//!
//! A file's layout flatbuffer holds one `Layout` table per layout node. These tests build footers
//! whose root is a chunked layout over `chunks` flat layouts, so the layout flatbuffer holds
//! `chunks + 1` tables.

use std::sync::Arc;

use flatbuffers::InvalidFlatbuffer;
use flatbuffers::VerifierOptions;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_array::flatbuffers::FILE_METADATA_MAX_DEPTH;
use vortex_array::flatbuffers::FILE_METADATA_MAX_TABLES;
use vortex_array::flatbuffers::FILE_METADATA_VERIFIER_OPTIONS;
use vortex_array::flatbuffers::WriteFlatBufferExt;
use vortex_array::flatbuffers::file_metadata_read_verifier_options;
use vortex_error::vortex_bail;
use vortex_layout::flatbuffers::layout as fbl;
use vortex_layout::LayoutContext;
use vortex_layout::LayoutRef;
use vortex_layout::layout_children;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::segments::SegmentId;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::DeserializeStep;
use crate::Footer;
use crate::footer::SegmentSpec;

fn dtype() -> DType {
    DType::Primitive(PType::I32, Nullability::NonNullable)
}

fn chunked_root(chunks: usize) -> VortexResult<LayoutRef> {
    let children = (0..chunks)
        .map(|chunk| {
            Ok(FlatLayout::new(
                1,
                dtype(),
                SegmentId::from(u32::try_from(chunk)?),
                ReadContext::new([]),
            )
            .into_layout())
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(
        ChunkedLayout::new(u64::try_from(chunks)?, dtype(), layout_children(children))
            .into_layout(),
    )
}

fn footer_with_flat_chunks(chunks: usize) -> VortexResult<Footer> {
    // Zero-length segments keep the file small; the reader only requires sorted offsets.
    let segments: Arc<[SegmentSpec]> = (0..chunks)
        .map(|_| SegmentSpec {
            offset: 0,
            length: 0,
            alignment: Alignment::none(),
        })
        .collect();
    Ok(Footer::new(
        chunked_root(chunks)?,
        segments,
        None,
        ReadContext::new([]),
    ))
}

fn write(footer: Footer, options: VerifierOptions) -> VortexResult<ByteBuffer> {
    let session = session();
    let buffers = footer
        .into_serializer(&session)
        .with_verifier_options(options)
        .serialize()?;
    let mut file = ByteBufferMut::with_capacity(buffers.iter().map(ByteBuffer::len).sum());
    for buffer in buffers {
        file.extend_from_slice(&buffer);
    }
    Ok(file.freeze())
}

fn session() -> VortexSession {
    VortexSession::empty().with::<LayoutSession>()
}

fn read(file: ByteBuffer) -> VortexResult<Footer> {
    let size = u64::try_from(file.len())?;
    match Footer::deserializer(file, session())
        .with_size(size)
        .deserialize()?
    {
        DeserializeStep::Done(footer) => Ok(footer),
        step => vortex_bail!("footer deserializer did not finish: {step:?}"),
    }
}

fn with_max_tables(max_tables: usize) -> VerifierOptions {
    VerifierOptions {
        max_tables,
        ..FILE_METADATA_VERIFIER_OPTIONS
    }
}

fn is_too_many_tables(error: &VortexError) -> bool {
    let mut source = std::error::Error::source(error);
    while let Some(error) = source {
        if let Some(flatbuffer) = error.downcast_ref::<InvalidFlatbuffer>() {
            return matches!(flatbuffer, InvalidFlatbuffer::TooManyTables);
        }
        source = error.source();
    }
    false
}

#[test]
fn more_than_a_million_layout_tables_round_trip() -> VortexResult<()> {
    const CHUNKS: usize = 1_000_000;
    let session = session();
    let layout = chunked_root(CHUNKS)?
        .flatbuffer_writer(&LayoutContext::default(), &session)
        .write_flatbuffer_bytes()?;
    // flatbuffers' default verifier, which readers used before, rejects this layout.
    let default_error = flatbuffers::root::<fbl::Layout>(&layout).unwrap_err();
    assert_eq!(default_error, InvalidFlatbuffer::TooManyTables);

    let footer = read(write(
        footer_with_flat_chunks(CHUNKS)?,
        FILE_METADATA_VERIFIER_OPTIONS,
    )?)?;
    assert_eq!(footer.segment_map().len(), CHUNKS);
    assert_eq!(footer.layout().nchildren(), CHUNKS);
    assert_eq!(footer.layout().row_count(), u64::try_from(CHUNKS)?);
    Ok(())
}

#[test]
fn writer_accepts_layout_at_the_table_limit() -> VortexResult<()> {
    const CHUNKS: usize = 99;
    let file = write(
        footer_with_flat_chunks(CHUNKS)?,
        with_max_tables(CHUNKS + 1),
    )?;
    assert_eq!(read(file)?.layout().nchildren(), CHUNKS);
    Ok(())
}

#[test]
fn writer_rejects_layout_one_table_over_the_limit() -> VortexResult<()> {
    const CHUNKS: usize = 99;
    let error = write(footer_with_flat_chunks(CHUNKS)?, with_max_tables(CHUNKS)).unwrap_err();
    assert!(is_too_many_tables(&error), "{error}");
    assert!(
        error
            .to_string()
            .contains("readers would reject the layout"),
        "{error}"
    );
    Ok(())
}

#[test]
fn read_limits_are_never_stricter_than_write_limits() {
    let read = file_metadata_read_verifier_options();
    assert!(read.max_tables >= FILE_METADATA_MAX_TABLES);
    assert!(read.max_depth >= FILE_METADATA_MAX_DEPTH);
    assert_eq!(
        read.max_apparent_size,
        FILE_METADATA_VERIFIER_OPTIONS.max_apparent_size
    );
    // Every file a default-option reader could open stays readable.
    let default = VerifierOptions::default();
    assert!(read.max_tables >= default.max_tables);
    assert!(read.max_depth >= default.max_depth);
    assert!(read.max_apparent_size >= default.max_apparent_size);
}
