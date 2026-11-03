// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A fixture written two ways, for the supplied-aggregates tests.
//!
//! The *computed* way is the zoned writer as a compacting writer uses it today: each column is a
//! zoned layout over a compressing strategy over chunked flat layouts, so the writer computes the
//! zone partials and the chunk statistics and compresses each zone. The *supplied* way hands the
//! writer zones that are already compressed, with partials the caller computed, through
//! [`SuppliedZoneAggregates`]; the zoned layouts there sit directly over chunked flat layouts.
//!
//! Both start from the same rows: five zones of 4,096 rows except a 1,500-row last one, an `i64`
//! key, a nullable `f64` with NaNs and an all-null zone, a nullable string with empty strings, long
//! strings and one string no upper bound can represent, and a nullable list of structs.

#![allow(dead_code, reason = "each test binary uses part of the fixture")]

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::LazyLock;

use futures::StreamExt as _;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::root;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutChildType;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::layouts::zoned::Zoned;
use vortex_layout::layouts::zoned::writer::SuppliedZoneAggregates;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::layouts::zoned::writer::ZonedStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;
use vortex_zstd::Zstd;

use crate::common::enable_all_registered_array_encodings;

pub const ZONE_LEN: usize = 4096;
pub const ZONES: usize = 5;
pub const LAST_ZONE_LEN: usize = 1500;
pub const COLUMNS: [&str; 4] = ["id", "score", "name", "tags"];
/// The column the top-level string rule compresses with zstd.
pub const STRING_COLUMN: usize = 2;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);
    session
});

static ADAPTIVE: LazyLock<BtrBlocksCompressor> =
    LazyLock::new(|| BtrBlocksCompressorBuilder::default().with_compact().build());

pub fn zone_len(zone: usize) -> usize {
    if zone + 1 == ZONES {
        LAST_ZONE_LEN
    } else {
        ZONE_LEN
    }
}

fn zone_start(zone: usize) -> usize {
    zone * ZONE_LEN
}

/// The rows of one zone as plain values.
pub struct ZoneValues {
    pub id: Vec<i64>,
    pub score: Vec<Option<f64>>,
    pub name: Vec<Option<String>>,
    pub tags: Vec<Option<Vec<(u32, String)>>>,
}

pub fn zone_values(zone: usize) -> ZoneValues {
    let start = zone_start(zone);
    let rows = start..start + zone_len(zone);
    ZoneValues {
        id: rows.clone().map(|row| (row as i64) * 3).collect(),
        score: rows
            .clone()
            .map(|row| {
                if zone == 2 || row % 11 == 0 {
                    None
                } else if row % 13 == 0 {
                    Some(f64::NAN)
                } else {
                    Some((row % 1000) as f64 * 0.5 - 99.75)
                }
            })
            .collect(),
        name: rows
            .clone()
            .map(|row| {
                if row % 11 == 3 {
                    None
                } else if row % 17 == 0 {
                    Some(String::new())
                } else if zone == 3 && row == start + 7 {
                    // No upper bound of 64 bytes or fewer exists for this value.
                    Some("\u{10FFFF}".repeat(40))
                } else if row % 101 == 0 {
                    Some(format!("long-{row:08}-{}", "y".repeat(90)))
                } else {
                    Some(format!(
                        "user-{:05}-{}",
                        (row * 7919) % 10007,
                        "x".repeat(row % 13)
                    ))
                }
            })
            .collect(),
        tags: rows
            .map(|row| {
                (row % 19 != 0).then(|| {
                    (0..row % 4)
                        .map(|entry| {
                            (
                                u32::try_from((row + entry) % 37).vortex_expect("small key id"),
                                format!("v{}", (row * 31 + entry) % 997),
                            )
                        })
                        .collect()
                })
            })
            .collect(),
    }
}

/// The uncompressed columns of one zone, built fresh on every call so the two writes never share
/// an array or its cached statistics.
pub fn zone_columns(zone: usize) -> Vec<ArrayRef> {
    let values = zone_values(zone);

    let id = PrimitiveArray::from_iter(values.id.iter().copied()).into_array();
    let score = PrimitiveArray::from_option_iter(values.score.iter().copied()).into_array();
    let name = VarBinArray::from_iter(
        values.name.iter().map(Option::as_deref),
        DType::Utf8(Nullability::Nullable),
    )
    .into_array();

    let mut offsets = Vec::with_capacity(values.tags.len() + 1);
    let mut key_ids = Vec::new();
    let mut entry_values = Vec::new();
    offsets.push(0i32);
    for row in &values.tags {
        for (key_id, value) in row.iter().flatten() {
            key_ids.push(*key_id);
            entry_values.push(value.as_str());
        }
        offsets.push(i32::try_from(key_ids.len()).vortex_expect("small offset"));
    }
    let entries = StructArray::try_new(
        FieldNames::from(["key_id", "value"]),
        vec![
            PrimitiveArray::from_iter(key_ids.iter().copied()).into_array(),
            VarBinArray::from_iter_nonnull(entry_values, DType::Utf8(Nullability::NonNullable))
                .into_array(),
        ],
        key_ids.len(),
        Validity::NonNullable,
    )
    .vortex_expect("entries")
    .into_array();
    let tags = ListArray::try_new(
        entries,
        PrimitiveArray::from_iter(offsets).into_array(),
        Validity::from_iter(values.tags.iter().map(Option::is_some)),
    )
    .vortex_expect("tags")
    .into_array();

    vec![id, score, name, tags]
}

pub fn zone_struct(fields: Vec<ArrayRef>, len: usize) -> ArrayRef {
    StructArray::try_new(
        FieldNames::from(COLUMNS),
        fields,
        len,
        Validity::NonNullable,
    )
    .vortex_expect("zone struct")
    .into_array()
}

/// The codec a compacting writer runs per column chunk: top-level strings to zstd level 3 in
/// 8,192-value frames without a dictionary, everything else to the compact adaptive compressor.
pub fn compress(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    if array.dtype().is_utf8() || array.dtype().is_binary() {
        let views = array.clone().execute::<VarBinViewArray>(ctx)?;
        Zstd::from_var_bin_view_without_dict(&views, 3, 8_192, ctx).map(IntoArray::into_array)
    } else {
        ADAPTIVE.compress(array, ctx)
    }
}

fn zoned_options() -> ZonedLayoutOptions {
    ZonedLayoutOptions {
        block_size: NonZeroUsize::new(ZONE_LEN).vortex_expect("non-zero zone"),
        aggregate_fns: None,
        concurrency: NonZeroUsize::new(1).vortex_expect("one"),
    }
}

/// Today's strategy: the writer computes zone partials and chunk statistics, then compresses.
pub fn computed_strategy() -> Arc<dyn LayoutStrategy> {
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let mut table = TableStrategy::new(Arc::clone(&flat), Arc::clone(&flat));
    for name in COLUMNS {
        let compressed = CompressingStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            compress,
        )
        .with_concurrency(1);
        let zoned = ZonedStrategy::new(compressed, FlatLayoutStrategy::default(), zoned_options());
        table = table.with_field_writer(FieldPath::from_name(name), Arc::new(zoned));
    }
    Arc::new(table)
}

/// A strategy whose zoned layouts take each column's partials from `supplied` and whose chunks
/// arrive compressed.
pub fn supplied_strategy(supplied: &[SuppliedZoneAggregates]) -> Arc<dyn LayoutStrategy> {
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let mut table = TableStrategy::new(Arc::clone(&flat), Arc::clone(&flat));
    for (name, supplied) in COLUMNS.iter().zip(supplied) {
        let zoned = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            zoned_options(),
        )
        .with_supplied_aggregates(supplied.clone());
        table = table.with_field_writer(FieldPath::from_name(*name), Arc::new(zoned));
    }
    Arc::new(table)
}

/// A strategy whose zoned layouts compute partials but whose chunks arrive compressed: the
/// writer without the supplied path, handed prepared chunks.
pub fn uncompressing_strategy() -> Arc<dyn LayoutStrategy> {
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let mut table = TableStrategy::new(Arc::clone(&flat), Arc::clone(&flat));
    for name in COLUMNS {
        let zoned = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            zoned_options(),
        );
        table = table.with_field_writer(FieldPath::from_name(name), Arc::new(zoned));
    }
    Arc::new(table)
}

/// The aggregates each column's zoned layout records, in the order its partials are supplied.
pub fn aggregate_fns_per_column() -> Vec<Arc<[AggregateFnRef]>> {
    let probe = ZonedStrategy::new(
        FlatLayoutStrategy::default(),
        FlatLayoutStrategy::default(),
        zoned_options(),
    );
    zone_columns(0)
        .iter()
        .map(|column| probe.aggregate_fns_for(column.dtype(), &SESSION))
        .collect()
}

/// One zone as an encoder hands it to the writer.
pub struct PreparedZone {
    pub len: usize,
    /// The zone's columns, each compressed.
    pub columns: Vec<ArrayRef>,
    /// Per column, the zone partials in [`aggregate_fns_per_column`] order.
    pub partials: Vec<Vec<Scalar>>,
}

/// Prepare a zone the way the computed path treats it, but outside the writer: the partials the
/// zoned layout would compute, the chunk statistics the compressing strategy computes before it
/// compresses, then the same codec.
///
/// With `inherit_stats` the compressed chunk also carries the uncompressed chunk's exact
/// statistics, which the computed path does not do; it shows what the zoned writer does when it
/// computes partials from a compressed chunk that has them.
pub fn prepare_zone(
    zone: usize,
    aggregate_fns: &[Arc<[AggregateFnRef]>],
    inherit_stats: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PreparedZone> {
    let all_stats = Stat::all().collect::<Vec<_>>();
    let mut columns = Vec::with_capacity(COLUMNS.len());
    let mut partials = Vec::with_capacity(COLUMNS.len());
    for (column, fns) in zone_columns(zone).into_iter().zip(aggregate_fns) {
        partials.push(
            fns.iter()
                .map(|aggregate_fn| {
                    let mut accumulator = aggregate_fn.accumulator(column.dtype())?;
                    accumulator.accumulate(&column, ctx)?;
                    accumulator.partial_scalar()
                })
                .collect::<VortexResult<Vec<_>>>()?,
        );
        column.statistics().compute_all(&all_stats, ctx)?;
        let compressed = compress(&column, ctx)?;
        if inherit_stats {
            compressed.statistics().inherit_from(column.statistics());
        }
        columns.push(compressed);
    }
    Ok(PreparedZone {
        len: zone_len(zone),
        columns,
        partials,
    })
}

pub fn prepare_zones(zones: std::ops::Range<usize>) -> VortexResult<Vec<PreparedZone>> {
    prepare_zones_with(zones, false)
}

pub fn prepare_zones_with(
    zones: std::ops::Range<usize>,
    inherit_stats: bool,
) -> VortexResult<Vec<PreparedZone>> {
    let aggregate_fns = aggregate_fns_per_column();
    let mut ctx = SESSION.create_execution_ctx();
    zones
        .map(|zone| prepare_zone(zone, &aggregate_fns, inherit_stats, &mut ctx))
        .collect()
}

pub fn new_supplied() -> Vec<SuppliedZoneAggregates> {
    COLUMNS
        .iter()
        .map(|_| SuppliedZoneAggregates::default())
        .collect()
}

pub fn options(strategy: Arc<dyn LayoutStrategy>) -> vortex_file::VortexWriteOptions {
    SESSION
        .write_options()
        .with_strategy(strategy)
        .with_file_statistics(Vec::new())
}

/// The file's struct dtype. Building it from arrays runs the list offsets' validation, which
/// computes statistics, so a probing test forces it before it starts the probe.
static STRUCT_DTYPE: LazyLock<DType> = LazyLock::new(|| {
    zone_struct(zone_columns(ZONES - 1), LAST_ZONE_LEN)
        .dtype()
        .clone()
});

pub fn struct_dtype() -> DType {
    STRUCT_DTYPE.clone()
}

fn computed_stream(zones: std::ops::Range<usize>) -> impl vortex_array::stream::ArrayStream {
    ArrayStreamAdapter::new(
        struct_dtype(),
        futures::stream::iter(
            zones.map(|zone| Ok(zone_struct(zone_columns(zone), zone_len(zone)))),
        ),
    )
}

/// The stream of prepared zones; each zone's partials are supplied just before the zone is
/// yielded, as an encoder running ahead of the writer would.
fn supplied_stream(
    prepared: Vec<PreparedZone>,
    supplied: Vec<SuppliedZoneAggregates>,
) -> impl vortex_array::stream::ArrayStream {
    ArrayStreamAdapter::new(
        struct_dtype(),
        futures::stream::iter(prepared).map(move |zone| {
            for (supplied, partials) in supplied.iter().zip(zone.partials) {
                supplied.push(zone.len, partials);
            }
            Ok(zone_struct(zone.columns, zone.len))
        }),
    )
}

pub fn compressed_stream(prepared: Vec<PreparedZone>) -> impl vortex_array::stream::ArrayStream {
    ArrayStreamAdapter::new(
        struct_dtype(),
        futures::stream::iter(prepared).map(|zone| Ok(zone_struct(zone.columns, zone.len))),
    )
}

pub async fn write_computed() -> VortexResult<ByteBuffer> {
    let mut bytes = Vec::new();
    options(computed_strategy())
        .write(&mut bytes, computed_stream(0..ZONES))
        .await?;
    Ok(ByteBuffer::from(bytes))
}

pub async fn write_supplied(
    prepared: Vec<PreparedZone>,
    supplied: Vec<SuppliedZoneAggregates>,
) -> VortexResult<ByteBuffer> {
    let mut bytes = Vec::new();
    options(supplied_strategy(&supplied))
        .write(&mut bytes, supplied_stream(prepared, supplied.clone()))
        .await?;
    for (name, supplied) in COLUMNS.iter().zip(&supplied) {
        if supplied.pending() != 0 {
            vortex_bail!(
                "{name}: {} supplied zones were not taken",
                supplied.pending()
            );
        }
    }
    Ok(ByteBuffer::from(bytes))
}

/// The prepared (compressed) zones through a zoned writer that computes its own partials.
pub async fn write_uncompressing(prepared: Vec<PreparedZone>) -> VortexResult<ByteBuffer> {
    let mut bytes = Vec::new();
    options(uncompressing_strategy())
        .write(&mut bytes, compressed_stream(prepared))
        .await?;
    Ok(ByteBuffer::from(bytes))
}

/// The computed path written as consecutive ranges and stitched.
pub async fn write_computed_stitched(splits: &[usize]) -> VortexResult<ByteBuffer> {
    let mut ranges = Vec::new();
    for window in splits.windows(2) {
        ranges.push(
            options(computed_strategy())
                .write_range(computed_stream(window[0]..window[1]))
                .await?,
        );
    }
    let mut bytes = Vec::new();
    options(computed_strategy())
        .write_stitched(&mut bytes, ranges)
        .await?;
    Ok(ByteBuffer::from(bytes))
}

/// The supplied path written as consecutive ranges, each with its own strategy and supplied
/// partials, and stitched.
pub async fn write_supplied_stitched(
    mut prepared: Vec<PreparedZone>,
    splits: &[usize],
) -> VortexResult<ByteBuffer> {
    let mut ranges = Vec::new();
    for window in splits.windows(2).rev() {
        let tail = prepared.split_off(window[0]);
        let supplied = new_supplied();
        ranges.push((options(supplied_strategy(&supplied)), tail, supplied));
    }
    let mut written = Vec::new();
    for (options, zones, supplied) in ranges.into_iter().rev() {
        let range = options
            .write_range(supplied_stream(zones, supplied.clone()))
            .await?;
        for (name, supplied) in COLUMNS.iter().zip(&supplied) {
            if supplied.pending() != 0 {
                vortex_bail!(
                    "{name}: {} supplied zones were not taken",
                    supplied.pending()
                );
            }
        }
        written.push(range);
    }
    let mut bytes = Vec::new();
    options(supplied_strategy(&new_supplied()))
        .write_stitched(&mut bytes, written)
        .await?;
    Ok(ByteBuffer::from(bytes))
}

pub fn open(bytes: ByteBuffer) -> VortexResult<VortexFile> {
    SESSION.open_options().open_buffer(bytes)
}

/// The zoned layout of one top-level column.
pub fn column_layout(file: &VortexFile, name: &str) -> VortexResult<LayoutRef> {
    let root = file.footer().layout();
    for (child_type, child) in root.child_types().zip(root.children()?) {
        if let LayoutChildType::Field(field) = child_type
            && field.as_ref() == name
        {
            return Ok(child);
        }
    }
    vortex_bail!("no column {name} in {}", root.encoding_id())
}

/// What one column's zoned layout wrote.
#[derive(Debug, PartialEq, Eq)]
pub struct ColumnZones {
    pub nzones: usize,
    pub zone_len: usize,
    pub present_aggregates: Vec<String>,
    /// The bytes of every segment under the zone table.
    pub zone_table_segments: Vec<Vec<u8>>,
    /// The bytes of every segment under the data child, in order.
    pub data_segments: Vec<Vec<u8>>,
}

fn segments_under(
    file: &VortexFile,
    bytes: &[u8],
    layout: &LayoutRef,
    out: &mut Vec<Vec<u8>>,
) -> VortexResult<()> {
    for segment_id in layout.segment_ids() {
        let spec = &file.footer().segment_map()[*segment_id as usize];
        let start = usize::try_from(spec.offset)?;
        let end = start + usize::try_from(spec.length)?;
        out.push(bytes[start..end].to_vec());
    }
    for child in layout.children()? {
        segments_under(file, bytes, &child, out)?;
    }
    Ok(())
}

pub fn column_zones(bytes: &ByteBuffer, name: &str) -> VortexResult<ColumnZones> {
    let file = open(bytes.clone())?;
    let layout = column_layout(&file, name)?;
    let Some(zoned) = layout.as_opt::<Zoned>() else {
        vortex_bail!("{name} is a {} layout, not zoned", layout.encoding_id());
    };
    let children = layout.children()?;
    let mut zone_table_segments = Vec::new();
    segments_under(
        &file,
        bytes.as_slice(),
        &children[1],
        &mut zone_table_segments,
    )?;
    let mut data_segments = Vec::new();
    segments_under(&file, bytes.as_slice(), &children[0], &mut data_segments)?;
    Ok(ColumnZones {
        nzones: zoned.nzones(),
        zone_len: zoned.zone_len(),
        present_aggregates: zoned.present_aggregates().iter().cloned().collect(),
        zone_table_segments,
        data_segments,
    })
}

/// The decoded zone table of one column, one struct scalar per zone.
pub async fn zone_table(bytes: &ByteBuffer, name: &str) -> VortexResult<Vec<Scalar>> {
    let file = open(bytes.clone())?;
    let layout = column_layout(&file, name)?;
    let children = layout.children()?;
    let zones = &children[1];
    let reader = zones.new_reader(
        "zones".into(),
        file.segment_source(),
        &SESSION,
        &Default::default(),
    )?;
    let rows = zones.row_count();
    let table = reader
        .projection_evaluation(
            &(0..rows),
            &root().bind(reader.dtype())?,
            MaskFuture::new_true(usize::try_from(rows)?),
        )?
        .await?;
    let mut ctx = SESSION.create_execution_ctx();
    (0..table.len())
        .map(|row| table.execute_scalar(row, &mut ctx))
        .collect()
}

/// The decoded rows of the whole file, one struct scalar per row.
pub async fn decoded_rows(bytes: &ByteBuffer) -> VortexResult<Vec<Scalar>> {
    let file = open(bytes.clone())?;
    let mut stream = file.scan()?.into_stream()?;
    let mut ctx = SESSION.create_execution_ctx();
    let mut rows = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        for row in 0..chunk.len() {
            rows.push(chunk.execute_scalar(row, &mut ctx)?);
        }
    }
    Ok(rows)
}

/// The statistics serialized with the first flat chunk of one column, as a reader sees them.
pub async fn first_chunk_stats(bytes: &ByteBuffer, name: &str) -> VortexResult<String> {
    let file = open(bytes.clone())?;
    let layout = column_layout(&file, name)?;
    let mut data = layout.children()?.remove(0);
    while data.nchildren() > 0 {
        data = data.children()?.remove(0);
    }
    let reader = data.new_reader(
        "chunk".into(),
        file.segment_source(),
        &SESSION,
        &Default::default(),
    )?;
    let rows = data.row_count();
    let array = reader
        .projection_evaluation(
            &(0..rows),
            &root().bind(reader.dtype())?,
            MaskFuture::new_true(usize::try_from(rows)?),
        )?
        .await?;
    Ok(format!(
        "{} {:?}",
        array.encoding_id(),
        array.statistics().to_owned()
    ))
}

/// The first byte offset at which two buffers differ, if any.
pub fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then(|| left.len().min(right.len())))
}
