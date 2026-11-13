// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Uncached visits over the verified layout view. Unmodelled encodings use the normal build.
use vortex_array::DeserializeMetadata;
use vortex_array::EmptyMetadata;
use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_flatbuffers::layout as fbl;
use vortex_session::registry::ReadContext;

use crate::DynLayout;
use crate::LayoutChildType;
use crate::LayoutId;
use crate::children::ViewedLayoutChildren;
use crate::layouts::chunked::Chunked;
use crate::layouts::flat::Flat;
use crate::layouts::flat::FlatLayoutMetadata;
use crate::segments::SegmentId;

/// Controls a pre-order visit. Returning false from enter skips that node's children.
pub trait LayoutVisitor {
    fn enter(&mut self, node: &VisitedLayout<'_>) -> VortexResult<bool>;
    fn leave(&mut self) {}
    fn wants_row_offsets(&self) -> bool {
        false
    }
}

/// Position relative to the parent; a visit root is always at offset zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisitedPosition {
    Root,
    Positioned(Option<u64>),
    Auxiliary,
}

/// Segment ids borrowed from either a built node or the verified flatbuffer.
#[derive(Clone, Copy)]
pub enum VisitedSegments<'a> {
    Materialised(&'a [SegmentId]),
    Flatbuffer(flatbuffers::Vector<'a, u32>),
}
impl VisitedSegments<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::Materialised(ids) => ids.len(),
            Self::Flatbuffer(ids) => ids.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, idx: usize) -> SegmentId {
        match self {
            Self::Materialised(ids) => ids[idx],
            Self::Flatbuffer(ids) => ids.get(idx).into(),
        }
    }
}

/// Borrowed facts about one node; no layout object is needed for built-in leaf/chunk views.
pub struct VisitedLayout<'a> {
    pub encoding: LayoutId,
    pub row_count: u64,
    pub segment_ids: VisitedSegments<'a>,
    pub position: VisitedPosition,
    pub materialised: bool,
}

/// Small per-visit totals. No process-wide atomic or per-node event on the visit path.
#[derive(Default)]
pub struct VisitStats {
    pub flatbuffer_nodes: u64,
    pub materialised_nodes: u64,
    pub batch_children: u64,
    pub fallbacks: u64,
    /// Bounded encoding attribution; additional distinct encodings count in fallbacks only.
    pub fallback_encodings: [Option<(LayoutId, u64)>; 16],
}

#[derive(Clone, Copy, Default)]
enum Kind {
    Flat,
    Chunked,
    #[default]
    Other,
}

/// One context per top-level visit, shared by the batch loop and all descendants.
pub struct VisitCx<'a> {
    visitor: &'a mut dyn LayoutVisitor,
    pub(crate) stats: &'a mut VisitStats,
    pub(crate) position: VisitedPosition,
    read_context: Option<ReadContext>,
    kinds: [Kind; 16],
    overflow_kinds: Vec<Kind>,
    row_offsets: bool,
}

impl<'a> VisitCx<'a> {
    pub(crate) fn new(visitor: &'a mut dyn LayoutVisitor, stats: &'a mut VisitStats) -> Self {
        let row_offsets = visitor.wants_row_offsets();
        Self {
            visitor,
            stats,
            position: VisitedPosition::Root,
            read_context: None,
            kinds: [Kind::Other; 16],
            overflow_kinds: Vec::new(),
            row_offsets,
        }
    }
    pub(crate) fn row_offsets(&self) -> bool {
        self.row_offsets
    }
    pub(crate) fn enter(&mut self, node: &VisitedLayout<'_>) -> VortexResult<bool> {
        self.visitor.enter(node)
    }
    pub(crate) fn leave(&mut self) {
        self.visitor.leave();
    }
    pub(crate) fn materialised(&mut self, node: &dyn DynLayout) -> VortexResult<()> {
        node.dyn_visit(self)
    }

    pub(crate) fn enter_materialised<T: DynLayout + ?Sized>(
        &mut self,
        node: &T,
        ids: &[SegmentId],
    ) -> VortexResult<()> {
        self.stats.materialised_nodes += 1;
        if self.enter(&VisitedLayout {
            encoding: node.dyn_encoding_id(),
            row_count: node.dyn_row_count(),
            segment_ids: VisitedSegments::Materialised(ids),
            position: self.position,
            materialised: true,
        })? {
            for slot in 0..node.dyn_nslots() {
                if let Some(child) = node.dyn_slot_uncached(slot)? {
                    self.position = match node.dyn_slot_type(slot) {
                        Some(LayoutChildType::Auxiliary(_)) => VisitedPosition::Auxiliary,
                        kind => VisitedPosition::Positioned(if self.row_offsets {
                            kind.and_then(|k| k.row_offset())
                        } else {
                            None
                        }),
                    };
                    self.materialised(child.as_ref())?;
                }
            }
        }
        self.leave();
        Ok(())
    }

    fn kind(&mut self, viewed: &ViewedLayoutChildren, index: u16) -> Kind {
        let current = viewed.read_context();
        if let Some(original) = &self.read_context {
            // A footer shares one layout_read_ctx backing across its viewed children. A
            // foreign adapter need not do so: mismatch uses today's build. Two empty id
            // slices can compare equal or not by allocator accident; either is safe:
            // an empty table classifies nothing, and mismatch only costs a fallback.
            if !std::ptr::eq(current.ids(), original.ids()) {
                return Kind::Other;
            }
        } else {
            if current.ids().len() > self.kinds.len() {
                self.overflow_kinds.reserve_exact(current.ids().len());
            }
            for (idx, id) in current.ids().iter().enumerate() {
                let kind = match viewed.registry().get(id) {
                    Some(encoding) if encoding.as_any().is::<Flat>() => Kind::Flat,
                    Some(encoding) if encoding.as_any().is::<Chunked>() => Kind::Chunked,
                    _ => Kind::Other,
                };
                if current.ids().len() > self.kinds.len() {
                    self.overflow_kinds.push(kind);
                } else {
                    self.kinds[idx] = kind;
                }
            }
            self.read_context = Some(current.clone());
        }
        if self.overflow_kinds.is_empty() {
            self.kinds.get(index as usize).copied().unwrap_or_default()
        } else {
            self.overflow_kinds
                .get(index as usize)
                .copied()
                .unwrap_or_default()
        }
    }

    pub(crate) fn flatbuffer(
        &mut self,
        viewed: &ViewedLayoutChildren,
        node: fbl::Layout<'_>,
        dtype: &DType,
    ) -> VortexResult<()> {
        let kind = self.kind(viewed, node.encoding());
        let metadata = node.metadata().map(|m| m.bytes()).unwrap_or_default();
        let segments = node.segments().unwrap_or_default();
        let children = node.children().unwrap_or_default();
        match kind {
            Kind::Flat => {
                let _metadata = ProstMetadata::<FlatLayoutMetadata>::deserialize(metadata)?;
                if segments.len() != 1 {
                    vortex_bail!("Flat layout must have exactly one segment ID");
                }
                if !children.is_empty() {
                    vortex_bail!("Flat layout must not have children");
                }
            }
            Kind::Chunked => {
                EmptyMetadata::deserialize(metadata)?;
                let mut rows = 0u64;
                for child in children {
                    rows = rows
                        .checked_add(child.row_count())
                        .ok_or_else(|| vortex_err!("Chunked child row counts overflow"))?;
                }
                if rows != node.row_count() {
                    vortex_bail!("Chunked child row counts do not add up to parent row count");
                }
            }
            Kind::Other => {
                self.stats.fallbacks += 1;
                if let Some(id) = viewed.read_context().resolve(node.encoding()) {
                    for entry in &mut self.stats.fallback_encodings {
                        match entry {
                            Some((existing, count)) if *existing == id => {
                                *count += 1;
                                break;
                            }
                            None => {
                                *entry = Some((id, 1));
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                return self.materialised(viewed.build_layout(node, dtype)?.as_ref());
            }
        }
        self.stats.flatbuffer_nodes += 1;
        // Resolved by classification above; the metadata checks have already run.
        let encoding = viewed
            .read_context()
            .resolve(node.encoding())
            .ok_or_else(|| vortex_err!("Unknown layout encoding index: {}", node.encoding()))?;
        let descend = self.enter(&VisitedLayout {
            encoding,
            row_count: node.row_count(),
            segment_ids: VisitedSegments::Flatbuffer(segments),
            position: self.position,
            materialised: false,
        })?;
        if descend && matches!(kind, Kind::Chunked) {
            let mut offset = 0;
            for child in children {
                self.position = VisitedPosition::Positioned(self.row_offsets.then_some(offset));
                self.flatbuffer(viewed, child, dtype)?;
                // Already checked the aggregate above, exactly as Chunked's offsets build.
                offset += child.row_count();
            }
        }
        self.leave();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
