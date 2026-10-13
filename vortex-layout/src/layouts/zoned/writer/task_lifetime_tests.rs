// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::task::Poll;

use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;

use super::ZonedLayoutOptions;
use super::ZonedStrategy;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::compressed::CompressingStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialArrayStreamExt;
use crate::session::LayoutSession;

fn owned_strategy(owner: Arc<dyn Send + Sync>) -> ZonedStrategy {
    ZonedStrategy::new(
        CompressingStrategy::new(
            FlatLayoutStrategy::default(),
            |array: &vortex_array::ArrayRef, _ctx: &mut vortex_array::ExecutionCtx| {
                Ok(array.clone())
            },
        )
        .with_concurrency(1),
        FlatLayoutStrategy::default(),
        ZonedLayoutOptions {
            concurrency: NonZeroUsize::MIN,
            ..Default::default()
        },
    )
    .with_task_lifetime(owner)
}

#[tokio::test]
async fn cancelled_queued_statistics_release_owner() -> VortexResult<()> {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with_tokio();
    let owner = Arc::new(());
    let weak = Arc::downgrade(&owner);
    let strategy = owned_strategy(owner);
    let (ptr, eof) = SequenceId::root().split();
    let stream = buffer![1_i32, 2, 3]
        .into_array()
        .to_array_stream()
        .sequenced(ptr);
    let mut write = Box::pin(strategy.write_stream(
        LayoutWriterContext::new(ArrayContext::empty()),
        Arc::new(TestSegments::default()),
        stream,
        eof,
        &session,
    ));
    // This current-thread runtime has not yielded to the queued CPU task.
    assert!(matches!(futures::poll!(write.as_mut()), Poll::Pending));
    drop(write);
    drop(strategy);
    // Cancellation is processed on the next runtime turn. The queued closure
    // must own the lifetime independently of both strategy and consumer.
    assert!(weak.upgrade().is_some());
    tokio::task::yield_now().await;
    assert!(weak.upgrade().is_none());
    Ok(())
}

#[tokio::test]
async fn completed_and_failed_writes_do_not_pin_owner_in_session_or_layout() -> VortexResult<()> {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with_tokio();
    for fail in [false, true] {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let strategy = owned_strategy(owner);
        let (ptr, eof) = SequenceId::root().split();
        let array = buffer![1_i32, 2, 3].into_array();
        let dtype = array.dtype().clone();
        let input = futures::stream::once(async move {
            if fail {
                Err(vortex_err!("input sentinel"))
            } else {
                Ok(array)
            }
        });
        let stream = ArrayStreamAdapter::new(dtype, input).sequenced(ptr);
        let result = strategy
            .write_stream(
                LayoutWriterContext::new(ArrayContext::empty()),
                Arc::new(TestSegments::default()),
                stream,
                eof,
                &session,
            )
            .await;
        drop(strategy);
        // Keep both the parent session and any returned layout/error alive.
        assert!(weak.upgrade().is_none());
        match result {
            Err(error) if fail => assert_eq!(error.to_string(), "Other error: input sentinel"),
            Ok(_) if !fail => {}
            _ => vortex_bail!("unexpected write outcome"),
        }
    }
    Ok(())
}
