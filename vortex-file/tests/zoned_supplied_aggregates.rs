// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A zoned writer handed compressed zones with supplied partials writes what the writer that
//! computes them writes.

#![cfg(feature = "zstd")]
#![expect(clippy::tests_outside_test_module)]
#![allow(clippy::use_debug, reason = "the tests print what differed")]

mod common;
mod supplied;

use futures::StreamExt as _;
use vortex_array::ArrayRef;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::bounded_max::make_bounded_max_partial_dtype;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::lower_bound;
use vortex_array::scalar::upper_bound;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::layouts::zoned::writer::SuppliedZoneAggregates;

use crate::supplied::*;

fn describe(label: &str, left: &ByteBuffer, right: &ByteBuffer) {
    eprintln!(
        "[zy615] {label}: {} vs {} bytes, equal {}, first difference {:?}",
        left.len(),
        right.len(),
        left.as_slice() == right.as_slice(),
        first_difference(left.as_slice(), right.as_slice())
    );
}

/// Compare every column's zone layout; print all of them before failing on any.
async fn assert_same_zones(
    label: &str,
    expected: &ByteBuffer,
    actual: &ByteBuffer,
) -> VortexResult<()> {
    let mut failures = Vec::new();
    for name in COLUMNS {
        let left = column_zones(expected, name)?;
        let right = column_zones(actual, name)?;
        let tables_equal = zone_table(expected, name).await? == zone_table(actual, name).await?;
        eprintln!(
            "[zy615] {label} {name}: nzones {}/{}, aggregates {:?}, zone table segments {} equal {}, decoded zone tables equal {tables_equal}, data segments {} equal {}",
            left.nzones,
            right.nzones,
            left.present_aggregates,
            left.zone_table_segments.len(),
            left.zone_table_segments == right.zone_table_segments,
            left.data_segments.len(),
            left.data_segments == right.data_segments,
        );
        if left.data_segments != right.data_segments {
            for (index, (l, r)) in left
                .data_segments
                .iter()
                .zip(&right.data_segments)
                .enumerate()
            {
                if l != r {
                    eprintln!(
                        "[zy615] {label} {name}: data segment {index}: {} vs {} bytes, first difference {:?}",
                        l.len(),
                        r.len(),
                        first_difference(l, r)
                    );
                }
            }
            eprintln!(
                "[zy615] {label} {name}: first chunk stats expected {}",
                first_chunk_stats(expected, name).await?
            );
            eprintln!(
                "[zy615] {label} {name}: first chunk stats actual   {}",
                first_chunk_stats(actual, name).await?
            );
        }
        if left.nzones != right.nzones
            || left.zone_len != right.zone_len
            || left.present_aggregates != right.present_aggregates
            || left.zone_table_segments != right.zone_table_segments
            || !tables_equal
        {
            failures.push(name);
        }
    }
    let rows_equal = decoded_rows(expected).await? == decoded_rows(actual).await?;
    eprintln!("[zy615] {label}: decoded rows equal {rows_equal}");
    if !failures.is_empty() || !rows_equal {
        vortex_bail!("{label}: zone layouts differ for {failures:?}, rows equal {rows_equal}");
    }
    Ok(())
}

#[tokio::test]
async fn supplied_aggregates_write_the_same_zone_tables() -> VortexResult<()> {
    let computed = write_computed().await?;
    let supplied = write_supplied(prepare_zones(0..ZONES)?, new_supplied()).await?;
    describe("serial computed vs supplied file", &computed, &supplied);
    for name in COLUMNS {
        eprintln!(
            "[zy615] computed {name} first chunk: {}",
            first_chunk_stats(&computed, name).await?
        );
    }
    assert_same_zones("serial", &computed, &supplied).await
}

#[tokio::test]
async fn supplied_aggregates_write_the_same_file_bytes() -> VortexResult<()> {
    let computed = write_computed().await?;
    let supplied = write_supplied(prepare_zones(0..ZONES)?, new_supplied()).await?;
    assert!(
        computed.as_slice() == supplied.as_slice(),
        "file bytes differ: {} vs {} bytes, first difference {:?}",
        computed.len(),
        supplied.len(),
        first_difference(computed.as_slice(), supplied.as_slice())
    );
    Ok(())
}

fn position<F: Fn(&AggregateFnRef) -> bool>(
    fns: &[AggregateFnRef],
    find: F,
) -> VortexResult<usize> {
    fns.iter()
        .position(find)
        .ok_or_else(|| vortex_err!("aggregate missing from {fns:?}"))
}

/// Supplied partials that disagree with the data land in the zone table unchanged: the writer
/// takes them, it does not recompute them.
#[tokio::test]
async fn supplied_partials_are_adopted_not_recomputed() -> VortexResult<()> {
    let aggregate_fns = aggregate_fns_per_column();
    let mut prepared = prepare_zones(0..ZONES)?;
    let min = position(&aggregate_fns[0], |f| f.is::<Min>())?;
    prepared[1].partials[0][min] = Scalar::primitive(-12_345i64, Nullability::Nullable);
    let bounded_max = position(&aggregate_fns[STRING_COLUMN], |f| f.is::<BoundedMax>())?;
    prepared[2].partials[STRING_COLUMN][bounded_max] = Scalar::struct_(
        make_bounded_max_partial_dtype(&DType::Utf8(Nullability::Nullable)),
        vec![
            Scalar::utf8("zzzz-supplied", Nullability::Nullable),
            Scalar::bool(false, Nullability::NonNullable),
        ],
    );

    let computed = write_computed().await?;
    let supplied = write_supplied(prepared, new_supplied()).await?;
    for (column, zone, marker) in [(0, 1, "-12345"), (STRING_COLUMN, 2, "zzzz-supplied")] {
        let name = COLUMNS[column];
        let expected = zone_table(&computed, name).await?;
        let actual = zone_table(&supplied, name).await?;
        eprintln!(
            "[zy615] adopted {name} zone {zone}: computed {} supplied {}",
            expected[zone], actual[zone]
        );
        for (index, (left, right)) in expected.iter().zip(&actual).enumerate() {
            if index == zone {
                assert_ne!(left, right, "{name} zone {zone} was recomputed");
                assert!(
                    right.to_string().contains(marker),
                    "{name} zone {zone} lacks the supplied {marker}: {right}"
                );
            } else {
                assert_eq!(left, right, "{name} zone {index}");
            }
        }
    }
    Ok(())
}

/// The partial a caller builds from plain values, with no Vortex compute, for each aggregate this
/// fixture records.
fn hand_partial(
    aggregate_fn: &AggregateFnRef,
    dtype: &DType,
    values: &ColumnValues<'_>,
) -> VortexResult<Scalar> {
    if aggregate_fn.is::<NullCount>() {
        return Ok(Scalar::primitive(values.nulls(), Nullability::NonNullable));
    }
    if aggregate_fn.is::<NanCount>() {
        let ColumnValues::Float(values) = values else {
            vortex_bail!("NaN count of a non-float column");
        };
        let nans = values
            .iter()
            .flatten()
            .filter(|value| value.is_nan())
            .count();
        return Ok(Scalar::primitive(nans as u64, Nullability::NonNullable));
    }
    if aggregate_fn.is::<Min>() || aggregate_fn.is::<Max>() {
        return hand_extremum(aggregate_fn.is::<Max>(), dtype, values);
    }
    let ColumnValues::Text(values) = values else {
        vortex_bail!("no hand-built partial for {aggregate_fn} on {dtype}");
    };
    if let Some(options) = aggregate_fn.as_opt::<BoundedMin>() {
        let min = values.iter().flatten().min();
        return Ok(lower_bound(
            min.map(|value| BufferString::from(value.as_str())),
            options.max_bytes.get(),
            Nullability::Nullable,
        )
        .map_or_else(|| Scalar::null(dtype.as_nullable()), |(bound, _)| bound));
    }
    if let Some(options) = aggregate_fn.as_opt::<BoundedMax>() {
        return Ok(hand_bounded_max(values, dtype, options.max_bytes.get()));
    }
    vortex_bail!("no hand-built partial for {aggregate_fn}")
}

/// The skip-NaN minimum or maximum, null when the zone has no value.
fn hand_extremum(max: bool, dtype: &DType, values: &ColumnValues<'_>) -> VortexResult<Scalar> {
    let extremum = match values {
        ColumnValues::Int(values) => {
            let value = if max {
                values.iter().max()
            } else {
                values.iter().min()
            };
            value.map(|value| Scalar::primitive(*value, Nullability::Nullable))
        }
        ColumnValues::Float(values) => {
            let finite = values
                .iter()
                .flatten()
                .copied()
                .filter(|value| !value.is_nan());
            let value = if max {
                finite.reduce(f64::max)
            } else {
                finite.reduce(f64::min)
            };
            value.map(|value| Scalar::primitive(value, Nullability::Nullable))
        }
        ColumnValues::Text(_) | ColumnValues::List(_) => {
            vortex_bail!("no plain extremum for {dtype}")
        }
    };
    Ok(extremum.unwrap_or_else(|| Scalar::null(dtype.as_nullable())))
}

/// The bounded maximum's partial: the shortest upper bound within `max_bytes`, or `unknown` when
/// no such bound exists.
fn hand_bounded_max(values: &[Option<String>], dtype: &DType, max_bytes: usize) -> Scalar {
    let partial_dtype = make_bounded_max_partial_dtype(dtype);
    let Some(max) = values.iter().flatten().max() else {
        return Scalar::null(partial_dtype);
    };
    let (bound, unknown) = match upper_bound(
        Some(BufferString::from(max.as_str())),
        max_bytes,
        Nullability::Nullable,
    ) {
        Some((bound, _)) => (bound, false),
        None => (Scalar::null(dtype.as_nullable()), true),
    };
    Scalar::struct_(
        partial_dtype,
        vec![bound, Scalar::bool(unknown, Nullability::NonNullable)],
    )
}

enum ColumnValues<'a> {
    Int(&'a [i64]),
    Float(&'a [Option<f64>]),
    Text(&'a [Option<String>]),
    List(&'a [Option<Vec<(u32, String)>>]),
}

impl ColumnValues<'_> {
    fn nulls(&self) -> u64 {
        let nulls = match self {
            Self::Int(_) => 0,
            Self::Float(values) => values.iter().filter(|value| value.is_none()).count(),
            Self::Text(values) => values.iter().filter(|value| value.is_none()).count(),
            Self::List(values) => values.iter().filter(|value| value.is_none()).count(),
        };
        nulls as u64
    }
}

/// An encoder that folds plain min, max and counts and builds the partials itself writes the
/// computed zone tables.
#[tokio::test]
async fn hand_built_partials_write_the_same_zone_tables() -> VortexResult<()> {
    let aggregate_fns = aggregate_fns_per_column();
    let mut prepared = prepare_zones(0..ZONES)?;
    let dtypes = zone_columns(0)
        .iter()
        .map(|column| column.dtype().clone())
        .collect::<Vec<_>>();
    let mut mismatches = 0;
    for (zone, prepared) in prepared.iter_mut().enumerate() {
        let values = zone_values(zone);
        let columns = [
            ColumnValues::Int(&values.id),
            ColumnValues::Float(&values.score),
            ColumnValues::Text(&values.name),
            ColumnValues::List(&values.tags),
        ];
        for (column, values) in columns.iter().enumerate() {
            let hand = aggregate_fns[column]
                .iter()
                .map(|aggregate_fn| hand_partial(aggregate_fn, &dtypes[column], values))
                .collect::<VortexResult<Vec<_>>>()?;
            for ((aggregate_fn, computed), hand) in aggregate_fns[column]
                .iter()
                .zip(&prepared.partials[column])
                .zip(&hand)
            {
                let partial_dtype = computed.dtype().as_nullable();
                if computed.cast(&partial_dtype)? != hand.cast(&partial_dtype)? {
                    mismatches += 1;
                    eprintln!(
                        "[zy615] hand partial zone {zone} {} {aggregate_fn}: computed {computed} hand {hand}",
                        COLUMNS[column]
                    );
                }
            }
            prepared.partials[column] = hand;
        }
    }
    eprintln!("[zy615] hand-built partials differing from computed: {mismatches}");
    let computed = write_computed().await?;
    let supplied = write_supplied(prepared, new_supplied()).await?;
    describe("serial computed vs hand-built file", &computed, &supplied);
    assert_same_zones("hand-built", &computed, &supplied).await
}

const SPLITS: [usize; 3] = [0, 2, ZONES];

/// Ranges written with supplied partials and stitched hold the serial computed write's zone
/// tables.
#[tokio::test]
async fn stitched_ranges_adopt_supplied_partials() -> VortexResult<()> {
    let serial = write_computed().await?;
    let computed_stitched = write_computed_stitched(&SPLITS).await?;
    let supplied_stitched = write_supplied_stitched(prepare_zones(0..ZONES)?, &SPLITS).await?;
    describe(
        "serial computed vs stitched computed",
        &serial,
        &computed_stitched,
    );
    describe(
        "serial computed vs stitched supplied",
        &serial,
        &supplied_stitched,
    );
    describe(
        "stitched computed vs stitched supplied",
        &computed_stitched,
        &supplied_stitched,
    );
    assert_same_zones("stitched", &serial, &supplied_stitched).await
}

#[tokio::test]
async fn stitched_supplied_ranges_write_the_serial_file_bytes() -> VortexResult<()> {
    let serial = write_computed().await?;
    let supplied_stitched = write_supplied_stitched(prepare_zones(0..ZONES)?, &SPLITS).await?;
    assert!(
        serial.as_slice() == supplied_stitched.as_slice(),
        "file bytes differ: {} vs {} bytes, first difference {:?}",
        serial.len(),
        supplied_stitched.len(),
        first_difference(serial.as_slice(), supplied_stitched.as_slice())
    );
    Ok(())
}

async fn write_with_supplied(
    supply: impl Fn(&[SuppliedZoneAggregates], PreparedZone) -> ArrayRef + Send + 'static,
) -> VortexResult<ByteBuffer> {
    let supplied = new_supplied();
    let prepared = prepare_zones(0..ZONES)?;
    let handles = supplied.clone();
    let stream = ArrayStreamAdapter::new(
        struct_dtype(),
        futures::stream::iter(prepared).map(move |zone| Ok(supply(&handles, zone))),
    );
    let mut bytes = Vec::new();
    options(supplied_strategy(&supplied))
        .write(&mut bytes, stream)
        .await?;
    Ok(ByteBuffer::from(bytes))
}

#[tokio::test]
async fn a_zone_without_supplied_partials_fails_the_write() {
    let error = write_with_supplied(|supplied, zone| {
        if zone.len == ZONE_LEN {
            for (supplied, partials) in supplied.iter().zip(zone.partials) {
                supplied.push(zone.len, partials);
            }
        }
        zone_struct(zone.columns, zone.len)
    })
    .await
    .expect_err("the last zone has no supplied partials");
    assert!(
        error.to_string().contains("before its supplied aggregates"),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn supplied_partials_for_another_row_count_fail_the_write() {
    let error = write_with_supplied(|supplied, zone| {
        for (supplied, partials) in supplied.iter().zip(zone.partials) {
            supplied.push(zone.len + 1, partials);
        }
        zone_struct(zone.columns, zone.len)
    })
    .await
    .expect_err("the supplied row count is wrong");
    assert!(
        error.to_string().contains("rows, the zone has"),
        "unexpected error: {error}"
    );
}
