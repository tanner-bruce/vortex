// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pre-allocation bounds for the packed-string dictionary helpers.

use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::BufferGrowthStrategy;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use super::sample::SAMPLE_COUNT;
use super::sample::SAMPLE_SIZE;

/// Conservative physical geometry for one string dictionary compression site.
///
/// Every count may be an upper bound. The caller establishes these facts before
/// allocation over primitive UInt32 codes, host Utf8/LargeUtf8 values with primitive
/// signed offsets, and trivial or ordinary non-nullable Bool bitmap validity.
/// Source arrays and backing remain the caller's separate reservation.
///
/// These facts also bound the referenced string-value child produced by dictionary
/// preparation. They do not describe arbitrary encodings or third-party schemes.
#[derive(Debug, Clone, Copy)]
pub struct StringDictionaryMemoryShape {
    /// Number of logical codes, including null and missing positions.
    pub code_count: u64,
    /// Number of retained input value slots, including unused and null slots.
    pub value_count: u64,
    /// Total bytes in the input value slots, including unreferenced values.
    pub value_bytes: u64,
    /// Sum of valid logical occurrences longer than the inline string limit.
    pub outlined_bytes: u64,
    /// Maximum byte length of a referenced value.
    pub max_value_length: u64,
    /// Whether logical nulls can occur, including null dictionary values.
    pub nullable: bool,
}

/// Bulk allocation capacities owned by the dictionary helper constructors.
///
/// These are phase bounds, not a complete compressor reservation. The caller must
/// compose simultaneous helper invocations, recursive schemes, fields, returned
/// segments and cancellation lifetimes. Generic codec workspace/output, statistics,
/// canonical conversions, serialization and bounded per-object bookkeeping are not
/// included. A caller must retain its existing treatment of those categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StringDictionaryCapacity {
    /// Seen-slot marker bytes and overlapping prefix hash tables during analysis.
    pub analysis_peak: u64,
    /// Translation/hash scratch and prepared dictionary construction capacity.
    pub prepare_peak: u64,
    /// Bulk backing retained by the prepared dictionary after scratch drops.
    pub prepare_returned: u64,
    /// Full occurrence pack construction, including descriptor transfer overlap.
    pub pack_peak: u64,
    /// Bulk backing retained by a completed full occurrence pack.
    pub pack_returned: u64,
    /// Simultaneous partition/count/range slabs in one string sampling call.
    pub sample_ranges_peak: u64,
}

impl StringDictionaryMemoryShape {
    /// Bounds the current `analyze_dictionary`, `prepare_dictionary`, `pack_ranges`
    /// and string-sampling constructors without scanning or allocating.
    ///
    /// The hash-table calculation follows pinned hashbrown 0.17.1. Descriptor Vec
    /// growth follows Rust 1.97/1.98's minimum capacity and doubling for single
    /// pushes. Revisit these terms if their constructors or dependencies change.
    ///
    /// # Errors
    /// Returns an error outside supported 64-bit/u32 statistics geometry, when a
    /// value cannot fit an outlined block, or when checked capacity arithmetic overflows.
    pub fn capacity(self) -> VortexResult<StringDictionaryCapacity> {
        vortex_ensure!(
            cfg!(target_pointer_width = "64"),
            "String dictionary capacity bounds require a 64-bit target"
        );
        vortex_ensure!(
            self.code_count <= u64::from(u32::MAX)
                && self.value_count <= u64::from(u32::MAX)
                && self.max_value_length < u64::from(u32::MAX),
            "String dictionary geometry exceeds the supported u32 domain"
        );

        let distinct = self.value_count.min(self.code_count);
        // The null prefix can be additional to every referenced non-null prefix.
        // During growth both old and new tables exist, each no larger than this.
        let analysis_peak = sum(&[
            self.value_count,
            product(2, hash_table_bytes(sum(&[distinct, 1])?, size_of::<u64>())?)?,
        ])?;

        let packed = packed_capacity(
            self.code_count,
            self.outlined_bytes,
            self.max_value_length,
            self.nullable,
        )?;
        let unique_bytes = self.value_bytes.min(self.outlined_bytes);
        let prepared = packed_capacity(
            distinct,
            unique_bytes,
            self.max_value_length.min(unique_bytes),
            false,
        )?;
        // Preparation creates the bitmap eagerly, even if normalization drops it.
        let prepared_codes = sum(&[
            product(self.code_count, size_of::<u32>() as u64)?,
            bitmap_bytes(self.code_count)?,
        ])?;
        let prepare_returned = sum(&[prepared_codes, prepared.returned])?;
        let prepare_peak = sum(&[
            prepared_codes,
            prepared.peak,
            product(self.value_count, size_of::<Option<u32>>() as u64)?,
            hash_table_bytes(distinct, size_of::<(&[u8], u32)>())?,
        ])?;

        Ok(StringDictionaryCapacity {
            analysis_peak,
            prepare_peak,
            prepare_returned,
            pack_peak: packed.peak,
            pack_returned: packed.returned,
            sample_ranges_peak: sampling_ranges_bytes(self.code_count)?,
        })
    }
}

/// Construction and returned capacity of one packed builder.
struct PackedCapacity {
    /// Payload plus overlapping descriptor allocations.
    peak: u64,
    /// Payload plus the final descriptor slab.
    returned: u64,
}

/// Counts all bulk builder allocations without per-object/control headers.
fn packed_capacity(
    rows: u64,
    outlined: u64,
    max_length: u64,
    nullable: bool,
) -> VortexResult<PackedCapacity> {
    let views = if rows == 0 {
        0
    } else {
        sum(&[
            product(rows, size_of::<BinaryView>() as u64)?,
            align_of::<BinaryView>() as u64,
        ])?
    };
    let validity = if nullable { bitmap_bytes(rows)? } else { 0 };
    let (body, blocks) = if outlined == 0 {
        (0, 0)
    } else {
        // Closed-block slack is less than the next value's bytes. The final
        // block is at most the default 2 MiB ceiling or one oversized value.
        let (initial, ceiling) = match BufferGrowthStrategy::default() {
            BufferGrowthStrategy::Exponential {
                current_size,
                max_size,
            } => (u64::from(current_size), u64::from(max_size)),
            BufferGrowthStrategy::Fixed { size } => (u64::from(size), u64::from(size)),
        };
        vortex_ensure!(
            initial > 0 && ceiling >= initial,
            "Invalid string buffer growth bounds"
        );
        let body = sum(&[product(2, outlined)?, max_length.max(ceiling), 1])?;
        // Each closed block's used bytes plus the next value exceed 4 KiB.
        // Summing counts each payload byte at most twice. Every outlined value
        // also has at least MAX_INLINED_SIZE+1 bytes.
        let blocks = rows
            .min(outlined / (BinaryView::MAX_INLINED_SIZE as u64 + 1))
            .min(sum(&[1, product(2, outlined)? / initial])?);
        (body, blocks)
    };
    let descriptor_bytes = size_of::<ByteBuffer>() as u64;
    let descriptor_capacity = if blocks == 0 {
        0
    } else {
        blocks
            .max(4)
            .checked_next_power_of_two()
            .ok_or_else(overflow)?
    };
    // First the completed ByteBuffer Vec moves to Arc. The array constructor then
    // keeps that Arc while converting through a BufferHandle Vec into another Arc.
    let handle_bytes = size_of::<BufferHandle>() as u64;
    let descriptor_peak = product(product(2, descriptor_capacity)?, descriptor_bytes)?.max(
        product(blocks, sum(&[descriptor_bytes, product(2, handle_bytes)?])?)?,
    );
    let payload = sum(&[views, validity, body])?;
    Ok(PackedCapacity {
        peak: sum(&[payload, descriptor_peak])?,
        returned: sum(&[payload, product(blocks, handle_bytes)?])?,
    })
}

/// The eagerly allocated ByteBufferMut uses the default preferred alignment.
fn bitmap_bytes(rows: u64) -> VortexResult<u64> {
    if rows == 0 {
        Ok(0)
    } else {
        sum(&[
            rows.div_ceil(8),
            Alignment::DEFAULT_ALIGNMENT.as_usize() as u64,
        ])
    }
}

/// Pinned hashbrown 0.17.1 raw table layout with conservative SIMD group alignment.
fn hash_table_bytes(capacity: u64, item_size: usize) -> VortexResult<u64> {
    if capacity == 0 {
        return Ok(0);
    }
    let buckets = match capacity {
        1..4 => 4,
        4..8 => 8,
        8..15 => 16,
        _ => (product(capacity, 8)? / 7)
            .checked_next_power_of_two()
            .ok_or_else(overflow)?,
    };
    let data = product(buckets, item_size as u64)?;
    let control_offset = product(sum(&[data, 15])? / 16, 16)?;
    sum(&[control_offset, buckets, 16])
}

/// Each of the three sampler Vecs fits this many 16-byte pair slots.
fn sampling_ranges_bytes(rows: u64) -> VortexResult<u64> {
    let sample_size = u64::from(SAMPLE_SIZE);
    let sample_count = u64::from(SAMPLE_COUNT);
    let runs = if rows <= product(sample_size, sample_count)? {
        1
    } else {
        product((rows / 100 / sample_size).div_ceil(16), 16)?.max(sample_count)
    };
    product(product(3, size_of::<(usize, usize)>() as u64)?, runs.max(4))
}

/// Adds allocation terms, rejecting arithmetic overflow.
fn sum(terms: &[u64]) -> VortexResult<u64> {
    terms.iter().try_fold(0_u64, |total, term| {
        total.checked_add(*term).ok_or_else(overflow)
    })
}

/// Multiplies allocation terms, rejecting arithmetic overflow.
fn product(left: u64, right: u64) -> VortexResult<u64> {
    left.checked_mul(right).ok_or_else(overflow)
}

/// Produces the shared checked-arithmetic error.
fn overflow() -> VortexError {
    vortex_err!("String dictionary capacity overflow")
}

#[cfg(test)]
mod tests {
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_buffer::BitBufferMut;
    use vortex_buffer::BufferAllocatorRef;
    use vortex_utils::aliases::hash_map::HashMap;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;

    fn shape() -> StringDictionaryMemoryShape {
        StringDictionaryMemoryShape {
            code_count: 16_384,
            value_count: 16,
            value_bytes: 16 * 4096,
            outlined_bytes: 16_384 * 4096,
            max_value_length: 4096,
            nullable: false,
        }
    }

    #[test]
    fn repeated_outlined_values_use_occurrence_bytes() -> VortexResult<()> {
        let input = shape();
        let capacity = input.capacity()?;
        assert!(capacity.pack_returned >= input.outlined_bytes);
        assert!(capacity.prepare_returned < capacity.pack_returned);
        assert!(capacity.prepare_peak >= capacity.prepare_returned);
        assert!(capacity.pack_peak >= capacity.pack_returned);
        Ok(())
    }

    #[test]
    fn unused_slots_remain_in_analysis_and_translation_scratch() -> VortexResult<()> {
        let input = StringDictionaryMemoryShape {
            code_count: 1024,
            value_count: 1_000_000,
            ..shape()
        };
        let capacity = input.capacity()?;
        assert!(capacity.analysis_peak >= input.value_count);
        assert!(capacity.prepare_peak >= input.value_count * size_of::<Option<u32>>() as u64);
        Ok(())
    }

    #[test]
    fn rejects_overflow_and_unsupported_value_lengths() {
        assert!(
            StringDictionaryMemoryShape {
                outlined_bytes: u64::MAX,
                ..shape()
            }
            .capacity()
            .is_err()
        );
        assert!(
            StringDictionaryMemoryShape {
                max_value_length: u64::from(u32::MAX),
                ..shape()
            }
            .capacity()
            .is_err()
        );
    }

    #[test]
    fn bitmap_bound_covers_constructor_rounding() -> VortexResult<()> {
        for rows in [0_usize, 1, 7, 8, 9, 255, 256, 257, 4096] {
            let buffer = BitBufferMut::with_capacity(rows);
            assert!(buffer.capacity() >= rows);
            assert!(buffer.capacity() as u64 <= bitmap_bytes(rows as u64)? * 8);
        }
        Ok(())
    }

    #[test]
    fn hash_constructor_capacity_fits_pinned_table_boundaries() -> VortexResult<()> {
        for requested in [
            0_usize, 1, 3, 4, 7, 8, 14, 15, 28, 29, 56, 57, 112, 113, 4096,
        ] {
            let prefixes = HashSet::<u64>::with_capacity(requested);
            let values = HashMap::<&[u8], u32>::with_capacity(requested);
            for (capacity, item_size) in [
                (prefixes.capacity(), size_of::<u64>()),
                (values.capacity(), size_of::<(&[u8], u32)>()),
            ] {
                assert!(capacity >= requested);
                assert_eq!(
                    hash_table_bytes(requested as u64, item_size)?,
                    hash_table_bytes(capacity as u64, item_size)?
                );
            }
        }
        Ok(())
    }

    #[test]
    fn prefix_growth_retains_room_for_old_and_new_tables() -> VortexResult<()> {
        let input = StringDictionaryMemoryShape {
            code_count: 4096,
            value_count: 4096,
            ..shape()
        };
        let bound = input.capacity()?.analysis_peak;
        let mut prefixes = HashSet::<u64>::with_capacity(2048);
        for prefix in 0..=4096 {
            let before = hash_table_bytes(prefixes.capacity() as u64, size_of::<u64>())?;
            prefixes.insert(prefix);
            let after = hash_table_bytes(prefixes.capacity() as u64, size_of::<u64>())?;
            assert!(input.value_count + before + after <= bound);
        }
        Ok(())
    }

    #[test]
    fn packed_constructor_blocks_and_nullable_rows_fit_geometry() -> VortexResult<()> {
        for length in [0, 12, 13, 4095, 4096, 8192, 2 * 1024 * 1024 + 1] {
            let value = vec![b'a'; length];
            let mut builder = VarBinViewBuilder::with_capacity_in(
                DType::Utf8(Nullability::Nullable),
                17,
                BufferAllocatorRef::statically_allocated(),
            );
            for _ in 0..16 {
                builder.append_value(&value);
            }
            builder.append_null();
            let result = builder.finish_into_varbinview();
            let outlined = if length > BinaryView::MAX_INLINED_SIZE {
                16 * length as u64
            } else {
                0
            };
            let bound = packed_capacity(17, outlined, length as u64, true)?;
            let visible = 17 * size_of::<BinaryView>() as u64
                + result
                    .data_buffers()
                    .iter()
                    .map(|buffer| buffer.len() as u64)
                    .sum::<u64>()
                + result.data_buffers().len() as u64 * size_of::<BufferHandle>() as u64;
            assert!(visible <= bound.returned);
            assert!(bound.returned <= bound.peak);
            assert_eq!(result.len(), 17);
        }
        Ok(())
    }

    #[test]
    fn sample_range_bound_tracks_actual_sampler_run_count() -> VortexResult<()> {
        for rows in [
            0,
            1024,
            1025,
            102_399,
            102_400,
            108_800,
            200_687,
            u32::MAX as usize,
        ] {
            let actual = if rows <= (SAMPLE_SIZE * SAMPLE_COUNT) as usize {
                1
            } else {
                super::super::sample::sample_count_approx_one_percent(rows) as u64
            };
            assert!(
                sampling_ranges_bytes(rows as u64)?
                    >= 3 * actual * size_of::<(usize, usize)>() as u64
            );
        }
        Ok(())
    }
}
