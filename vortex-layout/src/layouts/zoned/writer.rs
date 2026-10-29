//! Write-time assembly for zoned layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt as _;
use parking_lot::Mutex;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::zoned::AggregateStatsAccumulator;
use crate::layouts::zoned::ZonedLayout;
use crate::layouts::zoned::aggregate_partials;
use crate::layouts::zoned::schema::default_bounded_stat_max_bytes;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialArrayStreamExt;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Configuration for building zoned layouts.
///
/// The input stream is assumed to already be partitioned into one chunk per zone, except
/// possibly the final partial zone.
pub struct ZonedLayoutOptions {
    /// The size of a statistics block
    pub block_size: NonZeroUsize,
    /// The aggregate partials to collect for each block.
    ///
    /// If unset, the writer chooses pruning aggregates from the input dtype.
    pub aggregate_fns: Option<Arc<[AggregateFnRef]>>,
    /// Number of chunks to compute aggregate partials in parallel.
    pub concurrency: NonZeroUsize,
}

impl Default for ZonedLayoutOptions {
    fn default() -> Self {
        Self {
            block_size: unsafe { NonZeroUsize::new_unchecked(8192) },
            aggregate_fns: None,
            concurrency: unsafe {
                NonZeroUsize::new_unchecked(get_available_parallelism().unwrap_or(1))
            },
        }
    }
}

pub struct ZonedStrategy {
    child: Arc<dyn LayoutStrategy>,
    stats: Arc<dyn LayoutStrategy>,
    options: ZonedLayoutOptions,
    task_lifetime: Option<Arc<dyn Send + Sync>>,
}

impl ZonedStrategy {
    /// Create a writer that emits a data child plus an auxiliary per-zone stats child.
    pub fn new<Child: LayoutStrategy, Stats: LayoutStrategy>(
        child: Child,
        stats: Stats,
        options: ZonedLayoutOptions,
    ) -> Self {
        Self {
            child: Arc::new(child),
            stats: Arc::new(stats),
            options,
            task_lifetime: None,
        }
    }

    /// Retain an owner while this strategy or any of its statistics tasks is alive.
    ///
    /// Each CPU task captures a clone before scheduling. Cancelling the consuming
    /// future cannot release the owner while started statistics work is still
    /// running; discarded queued tasks release their clone without doing the work.
    /// The owner is not stored in the session or the resulting layout.
    pub fn with_task_lifetime(mut self, owner: Arc<dyn Send + Sync>) -> Self {
        self.task_lifetime = Some(owner);
        self
    }
}

#[async_trait]
impl LayoutStrategy for ZonedStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let requested_fns = self
            .options
            .aggregate_fns
            .clone()
            .unwrap_or_else(|| default_zoned_aggregate_fns(stream.dtype(), session));
        let compute_session = session.clone();

        let stats_accumulator = Arc::new(Mutex::new(AggregateStatsAccumulator::new(
            stream.dtype(),
            &requested_fns,
        )));
        // The accumulator has dropped the aggregates this dtype cannot hold, leaving the ones
        // this write would record. An aggregate the context forbids fails the write, like a
        // forbidden array or layout: dropping it silently would leave a file that prunes worse
        // than the caller asked for, with nothing in the output saying so.
        let aggregate_fns = stats_accumulator.lock().aggregate_fns();
        for aggregate_fn in aggregate_fns.iter() {
            if !ctx.allows_aggregate(&aggregate_fn.id()) {
                vortex_bail!("Aggregate {} not permitted by ctx", aggregate_fn.id());
            }
        }

        if let Some(deferral) = ctx.zone_deferral() {
            return self
                .write_deferred(
                    ctx.clone(),
                    deferral.clone(),
                    segment_sink,
                    stream,
                    eof,
                    session,
                    requested_fns,
                    aggregate_fns,
                )
                .await;
        }

        let stream_dtype = stream.dtype().clone();
        let concurrency = self.options.concurrency.get();
        let task_lifetime = self.task_lifetime.clone();
        let stream = stream
            .map(move |item| {
                let aggregate_fns = Arc::clone(&aggregate_fns);
                let session = compute_session.clone();
                let task_lifetime = task_lifetime.clone();
                session.handle().spawn_cpu(move || {
                    // This guard must survive both normal return and unwinding,
                    // independently of the strategy and the task's consumer.
                    let _task_lifetime = task_lifetime;
                    let (sequence_id, chunk) = item?;
                    let partials = aggregate_partials(
                        &chunk,
                        &aggregate_fns,
                        &mut session.create_execution_ctx(),
                    )?;
                    Ok::<_, VortexError>((sequence_id, chunk, partials))
                })
            })
            .buffered(concurrency);

        // Accumulate zone stats in stream order so the auxiliary table stays aligned with the
        // data child.
        let stats_accumulator2 = Arc::clone(&stats_accumulator);
        let stream = SequentialStreamAdapter::new(
            stream_dtype,
            stream.map(move |item| {
                let (sequence_id, chunk, partials) = item?;
                stats_accumulator2.lock().push_partials(partials)?;
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();

        let block_size = self.options.block_size;

        // The eof used for the data child should appear _before_ our own stats tables.
        let data_eof = eof.split_off();
        let data_layout = self
            .child
            .write_stream(
                ctx.clone(),
                Arc::clone(&segment_sink),
                stream,
                data_eof,
                session,
            )
            .await?;

        let mut exec_ctx = session.create_execution_ctx();
        let Some((stats_array, aggregate_fns)) =
            stats_accumulator.lock().as_array(&mut exec_ctx)?
        else {
            // If we have no stats (e.g. the DType doesn't support them), then we just return the
            // child layout.
            return Ok(data_layout);
        };

        // We must defer creating the stats table LayoutWriter until now, because the DType of
        // the table depends on which stats were successfully computed.
        let stats_stream = stats_array
            .into_array()
            .to_array_stream()
            .sequenced(eof.split_off());
        let zones_layout = self
            .stats
            .write_stream(ctx, Arc::clone(&segment_sink), stats_stream, eof, session)
            .await?;

        Ok(
            ZonedLayout::try_new(data_layout, zones_layout, block_size, aggregate_fns)?
                .into_layout(),
        )
    }
}

impl ZonedStrategy {
    /// Write the data child and keep the per-zone partials in `deferral` instead of writing the
    /// zone table. The returned layout is the bare data child; [`write_deferred_zones`] wraps the
    /// stitched data child once every range's partials are known, because whether an aggregate
    /// is kept depends on all zones of the file.
    #[expect(
        clippy::too_many_arguments,
        reason = "cold path: once per column per range write; mirrors write_stream's arguments"
    )]
    async fn write_deferred(
        &self,
        ctx: LayoutWriterContext,
        deferral: ZoneDeferral,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
        requested_fns: Arc<[AggregateFnRef]>,
        aggregate_fns: Arc<[AggregateFnRef]>,
    ) -> VortexResult<LayoutRef> {
        let stream_dtype = stream.dtype().clone();
        let concurrency = self.options.concurrency.get();
        let task_lifetime = self.task_lifetime.clone();
        let compute_session = session.clone();
        let stream = stream
            .map(move |item| {
                let aggregate_fns = Arc::clone(&aggregate_fns);
                let session = compute_session.clone();
                let task_lifetime = task_lifetime.clone();
                session.handle().spawn_cpu(move || {
                    let _task_lifetime = task_lifetime;
                    let (sequence_id, chunk) = item?;
                    let partials = aggregate_partials(
                        &chunk,
                        &aggregate_fns,
                        &mut session.create_execution_ctx(),
                    )?;
                    Ok::<_, VortexError>((sequence_id, chunk, partials))
                })
            })
            .buffered(concurrency);

        let partials = Arc::new(Mutex::new(Vec::<Vec<Scalar>>::new()));
        let partials2 = Arc::clone(&partials);
        let stream = SequentialStreamAdapter::new(
            stream_dtype.clone(),
            stream.map(move |item| {
                let (sequence_id, chunk, chunk_partials) = item?;
                partials2.lock().push(chunk_partials);
                Ok((sequence_id, chunk))
            }),
        )
        .sendable();

        let data_eof = eof.split_off();
        let data_layout = self
            .child
            .write_stream(ctx, segment_sink, stream, data_eof, session)
            .await?;
        let partials = std::mem::take(&mut *partials.lock());
        deferral.0.lock().push(DeferredZones {
            data: Arc::clone(&data_layout),
            dtype: stream_dtype,
            requested_fns,
            zone_len: self.options.block_size,
            stats: Arc::clone(&self.stats),
            partials,
        });
        Ok(data_layout)
    }
}

/// The zone maps a range write deferred, in the order its zoned strategies finished.
///
/// Each entry is keyed by the data layout it belongs to; [`DeferredZones::data`] is the same
/// reference that sits in the range's layout tree.
#[derive(Clone, Default)]
pub struct ZoneDeferral(Arc<Mutex<Vec<DeferredZones>>>);

impl ZoneDeferral {
    /// Take every deferred zone map recorded so far.
    pub fn take(&self) -> Vec<DeferredZones> {
        std::mem::take(&mut *self.0.lock())
    }
}

/// One zoned strategy's data child and per-zone partials from one range write.
pub struct DeferredZones {
    data: LayoutRef,
    dtype: DType,
    requested_fns: Arc<[AggregateFnRef]>,
    zone_len: NonZeroUsize,
    stats: Arc<dyn LayoutStrategy>,
    partials: Vec<Vec<Scalar>>,
}

impl DeferredZones {
    /// The data child this zone map annotates, as it appears in the range's layout tree.
    pub fn data(&self) -> &LayoutRef {
        &self.data
    }

    /// Zones (chunks) recorded.
    pub fn nzones(&self) -> usize {
        self.partials.len()
    }
}

/// Build one zoned layout over `data` from the deferred partials of consecutive range writes.
///
/// The partials are replayed in order into one accumulator, exactly as a single write would have
/// pushed them, so the aggregates kept, the table and its segment are the single write's.
pub async fn write_deferred_zones(
    parts: Vec<DeferredZones>,
    data: LayoutRef,
    ctx: LayoutWriterContext,
    segment_sink: SegmentSinkRef,
    mut eof: SequencePointer,
    session: &VortexSession,
) -> VortexResult<LayoutRef> {
    let Some(first) = parts.first() else {
        vortex_bail!("a deferred zone map needs at least one range");
    };
    let dtype = first.dtype.clone();
    let requested_fns = Arc::clone(&first.requested_fns);
    let zone_len = first.zone_len;
    let stats = Arc::clone(&first.stats);
    vortex_ensure!(
        &dtype == data.dtype(),
        "deferred zone map dtype {} differs from the stitched data {}",
        dtype,
        data.dtype()
    );
    let mut accumulator = AggregateStatsAccumulator::new(&dtype, &requested_fns);
    for part in parts {
        vortex_ensure!(
            part.dtype == dtype
                && part.zone_len == zone_len
                && part.requested_fns.len() == requested_fns.len()
                && part
                    .requested_fns
                    .iter()
                    .zip(requested_fns.iter())
                    .all(|(left, right)| left.to_string() == right.to_string()),
            "range zone maps disagree on dtype, zone length or aggregates"
        );
        for partials in part.partials {
            accumulator.push_partials(partials)?;
        }
    }
    let mut exec_ctx = session.create_execution_ctx();
    let Some((stats_array, aggregate_fns)) = accumulator.as_array(&mut exec_ctx)? else {
        return Ok(data);
    };
    let stats_stream = stats_array
        .into_array()
        .to_array_stream()
        .sequenced(eof.split_off());
    let ctx = ctx.without_zone_deferral();
    let zones_layout = stats
        .write_stream(ctx, segment_sink, stats_stream, eof, session)
        .await?;
    Ok(ZonedLayout::try_new(data, zones_layout, zone_len, aggregate_fns)?.into_layout())
}

fn default_zoned_aggregate_fns(dtype: &DType, session: &VortexSession) -> Arc<[AggregateFnRef]> {
    let (max, min) = match dtype {
        DType::Utf8(_) | DType::Binary(_) => (
            BoundedMax.bind(BoundedMaxOptions {
                max_bytes: default_bounded_stat_max_bytes(),
            }),
            BoundedMin.bind(BoundedMinOptions {
                max_bytes: default_bounded_stat_max_bytes(),
            }),
        ),
        _ => (
            Max.bind(NumericalAggregateOpts::skip_nans()),
            Min.bind(NumericalAggregateOpts::skip_nans()),
        ),
    };

    // Sum is deliberately absent: zone maps exist to prune, and a zone sum prunes nothing.
    // Its semantics are also unsettled - null-on-empty was changed in #9113 and reverted in
    // #9324 - so it is not a stat to record in every zone of every file, let alone freeze
    // into an edition. File-level statistics still record `Stat::Sum` via `PRUNING_STATS`.
    let mut aggregate_fns = vec![
        max,
        min,
        NanCount.bind(EmptyOptions),
        NullCount.bind(EmptyOptions),
    ];

    // Stats from spatial extension types are discovered from the registry at runtime instead.
    aggregate_fns.extend(session.aggregate_fns().zone_stat_defaults(dtype));

    aggregate_fns.into()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::IntoArray;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
    use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::sum::Sum;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::zoned::Zoned;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;

    /// Write three zones of primitives through `ctx`, returning the aggregates the zoned
    /// layout recorded.
    fn write_zones(ctx: LayoutWriterContext) -> VortexResult<Vec<String>> {
        let strategy = ZonedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ZonedLayoutOptions {
                block_size: NonZeroUsize::new(3).vortex_expect("non zero"),
                ..Default::default()
            },
        );
        let (ptr, eof) = SequenceId::root().split();
        let stream = ChunkedArray::from_iter([
            buffer![1, 2, 3].into_array(),
            buffer![4, 5, 6].into_array(),
            buffer![7, 8, 9].into_array(),
        ])
        .into_array()
        .to_array_stream()
        .sequenced(ptr);

        let layout = block_on(|handle: Handle| async move {
            let session = vortex_array::array_session()
                .with::<LayoutSession>()
                .with::<RuntimeSession>()
                .with_handle(handle);
            strategy
                .write_stream(
                    ctx,
                    Arc::new(TestSegments::default()),
                    stream,
                    eof,
                    &session,
                )
                .await
        })?;

        Ok(layout
            .as_::<Zoned>()
            .aggregate_fns()
            .iter()
            .map(|aggregate_fn| aggregate_fn.id().to_string())
            .collect())
    }

    #[test]
    fn unrestricted_context_writes_the_default_aggregates() -> VortexResult<()> {
        let written = write_zones(LayoutWriterContext::new(ArrayContext::empty()))?;
        assert!(written.contains(&Min.id().to_string()));
        assert!(written.contains(&Max.id().to_string()));
        assert!(
            !written.contains(&Sum.id().to_string()),
            "wrote {written:?}"
        );
        Ok(())
    }

    #[test]
    fn a_permitted_set_covering_the_defaults_writes_them() -> VortexResult<()> {
        let ctx = LayoutWriterContext::new(ArrayContext::empty()).with_allowed_aggregates(
            HashSet::from_iter([Min.id(), Max.id(), NanCount.id(), NullCount.id()]),
        );
        assert!(write_zones(ctx)?.contains(&Max.id().to_string()));
        Ok(())
    }

    #[test]
    fn a_forbidden_aggregate_fails_the_write() {
        let ctx = LayoutWriterContext::new(ArrayContext::empty())
            .with_allowed_aggregates(HashSet::from_iter([Min.id()]));
        let error = write_zones(ctx).expect_err("the default aggregates are not all permitted");
        assert!(
            error.to_string().contains("not permitted by ctx"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn default_aggregates_bound_variable_length_min_max() {
        let aggregate_fns = default_zoned_aggregate_fns(
            &DType::Utf8(Nullability::NonNullable),
            &vortex_array::array_session(),
        );

        assert_eq!(
            aggregate_fns[0].as_::<BoundedMax>().max_bytes,
            default_bounded_stat_max_bytes()
        );
        assert_eq!(
            aggregate_fns[1].as_::<BoundedMin>().max_bytes,
            default_bounded_stat_max_bytes()
        );
    }

    #[test]
    fn default_aggregates_keep_fixed_width_min_max_exact() {
        let aggregate_fns =
            default_zoned_aggregate_fns(&PType::I32.into(), &vortex_array::array_session());

        assert!(aggregate_fns[0].is::<Max>());
        assert!(aggregate_fns[1].is::<Min>());
        assert!(aggregate_fns[2].is::<NanCount>());
    }

    /// Zone maps never carry a sum, whether or not the dtype could hold one.
    #[rstest]
    #[case::summable(PType::I32.into())]
    #[case::not_summable(DType::Extension(
        Timestamp::new(TimeUnit::Microseconds, Nullability::Nullable).erased(),
    ))]
    fn default_aggregates_never_record_sum(#[case] dtype: DType) {
        let aggregate_fns = default_zoned_aggregate_fns(&dtype, &vortex_array::array_session());

        assert!(
            aggregate_fns
                .iter()
                .all(|aggregate_fn| !aggregate_fn.is::<Sum>())
        );
    }
}
