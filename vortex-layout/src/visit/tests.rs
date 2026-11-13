// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use std::sync::Arc;

use flatbuffers::FlatBufferBuilder;
use flatbuffers::WIPOffset;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_flatbuffers::FlatBuffer;
use vortex_session::VortexSession;

use super::*;
use crate::Layout;
use crate::LayoutDeserializeArgs;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::layout_from_flatbuffer_with_options;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::struct_::Struct;
use crate::segments::SegmentSource;
use crate::session::LayoutSessionExt;

struct Raw {
    encoding: u16,
    rows: u64,
    metadata: Vec<u8>,
    ids: Vec<u32>,
    children: Vec<Raw>,
}
impl Raw {
    fn flat() -> Self {
        Self {
            encoding: 0,
            rows: 3,
            metadata: vec![],
            ids: vec![7],
            children: vec![],
        }
    }
    fn chunked(children: Vec<Self>, rows: u64) -> Self {
        Self {
            encoding: 1,
            rows,
            metadata: vec![],
            ids: vec![],
            children,
        }
    }
    fn write<'a>(&self, builder: &mut FlatBufferBuilder<'a>) -> WIPOffset<fbl::Layout<'a>> {
        let children = self
            .children
            .iter()
            .map(|c| c.write(builder))
            .collect::<Vec<_>>();
        let children = builder.create_vector(&children);
        let metadata = builder.create_vector(&self.metadata);
        let segments = builder.create_vector(&self.ids);
        fbl::Layout::create(
            builder,
            &fbl::LayoutArgs {
                encoding: self.encoding,
                row_count: self.rows,
                metadata: Some(metadata),
                segments: Some(segments),
                children: Some(children),
            },
        )
    }
    fn open(&self, session: &VortexSession, allow_unknown: bool) -> VortexResult<LayoutRef> {
        let mut builder = FlatBufferBuilder::new();
        let root = self.write(&mut builder);
        builder.finish(root, None);
        layout_from_flatbuffer_with_options(
            FlatBuffer::copy_from(builder.finished_data()),
            &DType::Bool(Nullability::NonNullable),
            &ReadContext::new([Flat.id(), Chunked.id(), "unknown.visit.test".into()]),
            &ReadContext::new([]),
            session,
            allow_unknown,
        )
    }
}

#[derive(Default)]
struct Facts(Vec<(LayoutId, u64, Vec<SegmentId>, VisitedPosition)>);
impl LayoutVisitor for Facts {
    fn enter(&mut self, node: &VisitedLayout<'_>) -> VortexResult<bool> {
        self.0.push((
            node.encoding,
            node.row_count,
            (0..node.segment_ids.len())
                .map(|i| node.segment_ids.get(i))
                .collect(),
            node.position,
        ));
        Ok(true)
    }
    fn wants_row_offsets(&self) -> bool {
        true
    }
}
fn reference(node: &LayoutRef, position: VisitedPosition, facts: &mut Facts) -> VortexResult<()> {
    facts.0.push((
        node.encoding_id(),
        node.row_count(),
        node.segment_ids(),
        position,
    ));
    for slot in 0..node.nslots() {
        if let Some(child) = node.slot(slot)? {
            let position = match node.slot_type(slot) {
                Some(LayoutChildType::Auxiliary(_)) => VisitedPosition::Auxiliary,
                kind => VisitedPosition::Positioned(kind.and_then(|k| k.row_offset())),
            };
            reference(&child, position, facts)?;
        }
    }
    Ok(())
}

#[test]
fn visit_and_uncached_slot_preserve_facts_without_filling_cache() -> VortexResult<()> {
    let session = crate::test::new_session();
    for children in [vec![], vec![Raw::flat()], vec![Raw::flat(), Raw::flat()]] {
        let rows = children.len() as u64 * 3;
        let raw = Raw::chunked(children, rows);
        let viewed = raw.open(&session, false)?;
        let mut actual = Facts::default();
        viewed.visit(&mut actual)?;
        assert_eq!(viewed.cached_children(), 0);
        if viewed.nchildren() > 0 {
            let _child = viewed.slot_uncached(0)?;
            assert_eq!(viewed.cached_children(), 0);
        }
        let mut expected = Facts::default();
        reference(
            &raw.open(&session, false)?,
            VisitedPosition::Root,
            &mut expected,
        )?;
        assert_eq!(actual.0, expected.0);
    }
    Ok(())
}

#[test]
fn raw_faults_fail_on_the_same_node_as_cached_build() -> VortexResult<()> {
    let session = crate::test::new_session();
    let mut cases = Vec::new();
    let mut zero = Raw::flat();
    zero.ids.clear();
    cases.push(zero);
    let mut two = Raw::flat();
    two.ids.push(8);
    cases.push(two);
    let mut child = Raw::flat();
    child.children.push(Raw::flat());
    cases.push(child);
    let mut metadata = Raw::flat();
    metadata.metadata = vec![0xff];
    cases.push(metadata);
    let mut metadata = Raw::chunked(vec![Raw::flat()], 3);
    metadata.metadata = vec![1];
    cases.push(metadata);
    cases.push(Raw::chunked(vec![Raw::flat()], 2));
    cases.push(Raw::chunked(vec![Raw::flat()], 4));
    let mut huge = Raw::flat();
    huge.rows = u64::MAX;
    cases.push(Raw::chunked(vec![huge, Raw::flat()], 3));
    let mut unknown = Raw::flat();
    unknown.encoding = 2;
    cases.push(unknown);
    let mut unknown_index = Raw::flat();
    unknown_index.encoding = 90;
    cases.push(unknown_index);
    for allow_unknown in [false, true] {
        for child in &cases {
            // The valid outer Chunked builds without examining the faulty child.
            let mut builder = FlatBufferBuilder::new();
            let child_offset = child.write(&mut builder);
            let children = builder.create_vector(&[child_offset]);
            let root = fbl::Layout::create(
                &mut builder,
                &fbl::LayoutArgs {
                    encoding: 1,
                    row_count: child.rows,
                    children: Some(children),
                    ..Default::default()
                },
            );
            builder.finish(root, None);
            let bytes = builder.finished_data().to_vec();
            let open = || {
                layout_from_flatbuffer_with_options(
                    FlatBuffer::copy_from(&bytes),
                    &DType::Bool(Nullability::NonNullable),
                    &ReadContext::new([Flat.id(), Chunked.id(), "unknown.visit.test".into()]),
                    &ReadContext::new([]),
                    &session,
                    allow_unknown,
                )
            };
            let mut actual = Facts::default();
            let visited = open()?.visit(&mut actual).map_err(|e| e.to_string());
            let mut expected = Facts::default();
            let built = reference(&open()?, VisitedPosition::Root, &mut expected)
                .map_err(|e| e.to_string());
            assert_eq!(
                visited, built,
                "encoding {}, metadata {:?}",
                child.encoding, child.metadata
            );
            assert_eq!(
                actual.0, expected.0,
                "same successfully entered nodes before the fault"
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ReplacementFlat;
impl VTable for ReplacementFlat {
    type LayoutData = ();
    type Metadata = EmptyMetadata;
    fn id(&self) -> LayoutId {
        Flat.id()
    }
    fn metadata(_: &Layout<Self>) -> EmptyMetadata {
        EmptyMetadata
    }
    fn deserialize(&self, _: &LayoutDeserializeArgs<'_>, _: &EmptyMetadata) -> VortexResult<()> {
        Ok(())
    }
    fn child_dtype(layout: &Layout<Self>, _: usize) -> VortexResult<DType> {
        Ok(layout.dtype().clone())
    }
    fn child_type(_: &Layout<Self>, _: usize) -> LayoutChildType {
        LayoutChildType::Transparent("test".into())
    }
    fn new_reader(
        _: &Layout<Self>,
        _: Arc<str>,
        _: Arc<dyn SegmentSource>,
        _: &VortexSession,
        _: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        Err(vortex_err!("test-only plugin"))
    }
}
#[test]
fn plugin_under_builtin_id_uses_build_fallback() -> VortexResult<()> {
    let session = crate::test::new_session();
    session
        .layouts()
        .register(&ReplacementFlat as &dyn crate::LayoutEncoding);
    let mut flat = Raw::flat();
    flat.ids.clear(); // Built-in Flat would reject this.
    let node = Raw::chunked(vec![flat], 3).open(&session, false)?;
    let mut stats = VisitStats::default();
    node.visit_with_stats(&mut Facts::default(), &mut stats)?;
    assert_eq!(stats.fallbacks, 1);
    assert_eq!(stats.flatbuffer_nodes, 0);
    assert_eq!(stats.materialised_nodes, 2);
    assert_eq!(node.cached_children(), 0);
    Ok(())
}

#[test]
fn rebuild_uncached_matches_parsed_root_and_leaves_original_cache_intact() -> VortexResult<()> {
    let session = crate::test::new_session();
    let dtype = DType::Struct(
        StructFields::new(
            ["value"].into(),
            vec![DType::Bool(Nullability::NonNullable)],
        ),
        Nullability::NonNullable,
    );
    let raw = Raw {
        encoding: 2,
        rows: 6,
        metadata: vec![],
        ids: vec![],
        children: vec![Raw::chunked(vec![Raw::flat(), Raw::flat()], 6)],
    };
    let mut builder = FlatBufferBuilder::new();
    let root = raw.write(&mut builder);
    builder.finish(root, None);
    let original = layout_from_flatbuffer_with_options(
        FlatBuffer::copy_from(builder.finished_data()),
        &dtype,
        &ReadContext::new([Flat.id(), Chunked.id(), Struct.id()]),
        &ReadContext::new([]),
        &session,
        false,
    )?;
    let mut expected = Facts::default();
    reference(&original, VisitedPosition::Root, &mut expected)?;
    assert_eq!(original.cached_children(), 1);
    let rebuilt = original.rebuild_uncached().expect("viewed root")?;
    assert_eq!(rebuilt.cached_children(), 0);
    assert_eq!(original.cached_children(), 1);
    assert!(!Arc::ptr_eq(&original, &rebuilt));
    assert_eq!(rebuilt.encoding_id(), original.encoding_id());
    assert_eq!(rebuilt.dtype(), original.dtype());
    assert_eq!(rebuilt.row_count(), original.row_count());
    assert_eq!(rebuilt.segment_ids(), original.segment_ids());
    assert_eq!(rebuilt.nchildren(), original.nchildren());
    let mut actual = Facts::default();
    reference(&rebuilt, VisitedPosition::Root, &mut actual)?;
    assert_eq!(actual.0, expected.0);
    Ok(())
}

#[test]
fn rebuild_uncached_returns_none_for_owned_children() -> VortexResult<()> {
    let session = crate::test::new_session();
    let layout = ChunkedLayout::new(
        3,
        DType::Bool(Nullability::NonNullable),
        crate::layout_children(vec![Raw::flat().open(&session, false)?]),
    )
    .into_layout();
    assert!(layout.rebuild_uncached().is_none());
    Ok(())
}
