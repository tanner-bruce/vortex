// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Counts heap allocations on the write path, per serialized array and per file segment.
//!
//! The test binary has one test, so no other test thread allocates while it counts.

#![expect(clippy::tests_outside_test_module)]

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::FieldNames;
use vortex_array::serde::SerializeOptions;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::segment::Segment;
use vortex_error::VortexResult;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

mod common;

use common::enable_all_registered_array_encodings;

struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded to `System`; the counter has no effect on the allocation.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller upholds `GlobalAlloc::dealloc`'s contract.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn count<R>(f: impl FnOnce() -> R) -> (R, usize) {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let result = f();
    COUNTING.store(false, Ordering::Relaxed);
    (result, ALLOCATIONS.load(Ordering::Relaxed))
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    enable_all_registered_array_encodings(&session);
    session
});

const OPTIONS: SerializeOptions = SerializeOptions {
    offset: 0,
    include_padding: true,
};

/// Compressed columns of the kinds a part holds: integers, low-cardinality strings and floats.
fn columns() -> VortexResult<Vec<(&'static str, ArrayRef)>> {
    let mut ctx = SESSION.create_execution_ctx();
    let compressor = BtrBlocksCompressor::from_session(&SESSION);
    let ints = PrimitiveArray::from_iter((0..8_192i64).map(|i| i * 7 + (i % 13))).into_array();
    let strings = VarBinArray::from(
        (0..8_192)
            .map(|i| ["GET", "POST", "PUT", "DELETE"][i % 4])
            .collect::<Vec<_>>(),
    )
    .into_array();
    let floats = PrimitiveArray::from_iter((0..8_192).map(|i| f64::from(i) * 0.25)).into_array();
    [("ints", ints), ("strings", strings), ("floats", floats)]
        .into_iter()
        .map(|(name, array)| {
            let compressed = compressor.compress(&array, &mut ctx)?;
            Ok((name, compressed))
        })
        .collect()
}

/// 64 chunks of two columns, written flat per chunk, so the file has 128 small data segments
/// and the count is dominated by the per-segment path rather than by compression.
fn file_data() -> VortexResult<ArrayRef> {
    let chunks = (0..64usize)
        .map(|seed| {
            let ids = PrimitiveArray::from_iter((0..1_024u64).map(|i| seed as u64 * 1_024 + i))
                .into_array();
            let names = VarBinArray::from(
                (0..1_024)
                    .map(|i| ["GET", "POST", "PUT", "DELETE"][(i + seed) % 4])
                    .collect::<Vec<_>>(),
            )
            .into_array();
            StructArray::new(
                FieldNames::from(["id", "method"]),
                vec![ids, names],
                1_024,
                Validity::NonNullable,
            )
            .into_array()
        })
        .collect::<Vec<_>>();
    let dtype = chunks[0].dtype().clone();
    Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
}

#[test]
fn allocations_on_the_write_path() {
    let ctx = ArrayContext::empty();
    let columns = columns().expect("columns");
    for (name, array) in &columns {
        // Warm up: intern the encodings into the context.
        drop(
            array
                .serialize(&ctx, &SESSION, &OPTIONS)
                .expect("serialize"),
        );

        let (buffers, serialize) = count(|| {
            array
                .serialize(&ctx, &SESSION, &OPTIONS)
                .expect("serialize")
        });
        let pieces = buffers.len();
        drop(buffers);

        let (segment, fresh) = count(|| {
            let mut segment = Segment::new();
            array
                .serialize_into(&ctx, &SESSION, &OPTIONS, &mut segment)
                .expect("serialize_into");
            segment
        });
        let mut segment = segment;
        segment.clear();
        let (segment, recycled) = count(|| {
            array
                .serialize_into(&ctx, &SESSION, &OPTIONS, &mut segment)
                .expect("serialize_into");
            segment.piece_list().len()
        });
        println!(
            "{{\"event\":\"zy461_allocations\",\"case\":\"serialize\",\"column\":\"{name}\",\"buffers\":{pieces},\"serialize\":{serialize},\"serialize_into_fresh\":{fresh},\"serialize_into_recycled\":{recycled},\"pieces\":{segment}}}"
        );
        assert!(
            recycled < serialize,
            "{name}: a recycled segment allocates less than serialize ({recycled} vs {serialize})"
        );
    }

    let data = file_data().expect("file data");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");
    let _enter = runtime.enter();
    let session = SESSION.clone().with_tokio();
    let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
    let strategy: Arc<dyn LayoutStrategy> = Arc::new(TableStrategy::new(
        Arc::clone(&flat),
        Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())),
    ));
    let write = || {
        session
            .write_options()
            .with_strategy(Arc::clone(&strategy))
            .with_file_statistics(Vec::new())
            .write(Vec::new(), data.to_array_stream())
    };
    // Warm up the session's lazily built state.
    runtime.block_on(write()).expect("write");
    let (summary, allocations) = count(|| runtime.block_on(write()).expect("write"));
    let segments = summary.footer().segment_map().len();
    println!(
        "{{\"event\":\"zy461_allocations\",\"case\":\"file_write\",\"segments\":{segments},\"allocations\":{allocations},\"per_segment\":{:.1}}}",
        allocations as f64 / segments as f64
    );
}
