// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Traits for reading and writing Vortex types as flatbuffers.

use flatbuffers::FlatBufferBuilder;
use flatbuffers::Follow;
use flatbuffers::InvalidFlatbuffer;
use flatbuffers::Verifiable;
use flatbuffers::VerifierOptions;
use flatbuffers::WIPOffset;
use flatbuffers::root;
use flatbuffers::root_with_opts;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ConstByteBuffer;
use vortex_error::VortexResult;

/// We define a const-aligned byte buffer for flatbuffers with 8-byte alignment.
///
/// This is based on the assumption that the maximum primitive type is 8 bytes.
/// See: <https://groups.google.com/g/flatbuffers/c/PSgQeWeTx_g>
pub type FlatBuffer = ConstByteBuffer<8>;

/// Maximum number of tables accepted in one Vortex file metadata flatbuffer.
///
/// A file's layout flatbuffer holds one `Layout` table per layout node, and a large file has about
/// one layout node per segment. flatbuffers' default of 1,000,000 tables therefore rejects files
/// from a few tens of GiB: a 23.9 GiB file with 1,015,386 layout tables failed to open with
/// "Too many tables". 2^24 accepts layouts sixteen times larger while keeping verification work
/// bounded.
pub const FILE_METADATA_MAX_TABLES: usize = 1 << 24;

/// Maximum table nesting depth accepted in one Vortex file metadata flatbuffer (flatbuffers'
/// default).
pub const FILE_METADATA_MAX_DEPTH: usize = 64;

/// Maximum verifier "apparent size" accepted in one Vortex file metadata flatbuffer (flatbuffers'
/// default, 2 GiB).
///
/// The verifier counts every byte it checks, so the apparent size of a layout flatbuffer is about
/// twice its length (about 58 bytes per `Layout` table). A layout with
/// [`FILE_METADATA_MAX_TABLES`] tables stays below this limit.
pub const FILE_METADATA_MAX_APPARENT_SIZE: usize = 1 << 31;

/// Verifier options for the metadata flatbuffers of a Vortex file: the postscript, dtype, layout,
/// footer and file statistics.
///
/// Writers verify each metadata flatbuffer they serialize with these options, so a writer fails
/// instead of producing a file that readers reject. Readers use
/// [`file_metadata_read_verifier_options`], which is never stricter than these options.
pub const FILE_METADATA_VERIFIER_OPTIONS: VerifierOptions = VerifierOptions {
    max_depth: FILE_METADATA_MAX_DEPTH,
    max_tables: FILE_METADATA_MAX_TABLES,
    max_apparent_size: FILE_METADATA_MAX_APPARENT_SIZE,
    ignore_missing_null_terminator: false,
};

/// Verifier options readers use for Vortex file metadata flatbuffers.
///
/// These are [`FILE_METADATA_VERIFIER_OPTIONS`], optionally raised (never lowered) by the
/// `VORTEX_MAX_LAYOUT_TABLES` and `VORTEX_MAX_LAYOUT_DEPTH` environment variables, which are read
/// once per process. Raising the read limits lets a process open files that other writers produced
/// with larger limits; lowering them could reject files this crate's writers accept, so values
/// below the defaults are ignored.
pub fn file_metadata_read_verifier_options() -> &'static VerifierOptions {
    static OPTIONS: std::sync::LazyLock<VerifierOptions> = std::sync::LazyLock::new(|| {
        let env = |name: &str| std::env::var(name).ok();
        VerifierOptions {
            max_tables: raised_limit(
                env("VORTEX_MAX_LAYOUT_TABLES").as_deref(),
                FILE_METADATA_MAX_TABLES,
            ),
            max_depth: raised_limit(
                env("VORTEX_MAX_LAYOUT_DEPTH").as_deref(),
                FILE_METADATA_MAX_DEPTH,
            ),
            ..FILE_METADATA_VERIFIER_OPTIONS
        }
    });
    &OPTIONS
}

/// `default`, or a configured `value` when it parses and is larger.
fn raised_limit(value: Option<&str>, default: usize) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .map_or(default, |value| value.max(default))
}

/// Marker trait for types that can be the root of a FlatBuffer.
pub trait FlatBufferRoot {}

/// Trait for reading a type from a FlatBuffer.
pub trait ReadFlatBuffer: Sized {
    /// The FlatBuffer type that this type can be read from.
    type Source<'a>: Verifiable + Follow<'a>;
    /// The error type returned when reading fails.
    type Error: From<InvalidFlatbuffer>;

    /// Reads this type from a FlatBuffer source.
    fn read_flatbuffer<'buf>(
        fb: &<Self::Source<'buf> as Follow<'buf>>::Inner,
    ) -> Result<Self, Self::Error>;

    /// Reads this type from bytes representing a FlatBuffer source.
    fn read_flatbuffer_bytes<'buf>(bytes: &'buf [u8]) -> Result<Self, Self::Error>
    where
        <Self as ReadFlatBuffer>::Source<'buf>: 'buf,
    {
        let fb = root::<Self::Source<'buf>>(bytes)?;
        Self::read_flatbuffer(&fb)
    }

    /// Reads this type from bytes representing a FlatBuffer source, verified with `opts`.
    fn read_flatbuffer_bytes_with_opts<'buf>(
        bytes: &'buf [u8],
        opts: &VerifierOptions,
    ) -> Result<Self, Self::Error>
    where
        <Self as ReadFlatBuffer>::Source<'buf>: 'buf,
    {
        let fb = root_with_opts::<Self::Source<'buf>>(opts, bytes)?;
        Self::read_flatbuffer(&fb)
    }
}

/// Trait for writing a type to a FlatBuffer.
pub trait WriteFlatBuffer {
    /// The FlatBuffer type that this type can be written to.
    type Target<'a>;

    /// Writes this type to a FlatBuffer builder.
    fn write_flatbuffer<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> VortexResult<WIPOffset<Self::Target<'fb>>>;
}

/// Extension trait for types that can be written as FlatBuffer root objects.
pub trait WriteFlatBufferExt: WriteFlatBuffer + FlatBufferRoot {
    /// Writes self as a FlatBuffer root object into a [`FlatBuffer`] byte buffer.
    fn write_flatbuffer_bytes(&self) -> VortexResult<FlatBuffer>;
}

impl<F: WriteFlatBuffer + FlatBufferRoot> WriteFlatBufferExt for F {
    fn write_flatbuffer_bytes(&self) -> VortexResult<FlatBuffer> {
        let mut fbb = FlatBufferBuilder::new();
        let root_offset = self.write_flatbuffer(&mut fbb)?;
        fbb.finish_minimal(root_offset);
        let (vec, start) = fbb.collapse();
        let end = vec.len();
        Ok(FlatBuffer::align_from(
            ByteBuffer::from(vec).slice(start..end),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::FILE_METADATA_MAX_TABLES;
    use super::raised_limit;

    #[test]
    fn configured_read_limits_only_raise_the_default() {
        let default = FILE_METADATA_MAX_TABLES;
        assert_eq!(raised_limit(None, default), default);
        assert_eq!(raised_limit(Some("not a number"), default), default);
        assert_eq!(raised_limit(Some("1000000"), default), default);
        assert_eq!(
            raised_limit(Some(&(default - 1).to_string()), default),
            default
        );
        assert_eq!(raised_limit(Some(&default.to_string()), default), default);
        assert_eq!(
            raised_limit(Some(&(default + 1).to_string()), default),
            default + 1
        );
    }
}
