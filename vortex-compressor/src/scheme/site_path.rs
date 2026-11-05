// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression sites: where in a top-level array's structure a leaf is compressed.

use std::fmt;

use super::SchemeId;
use super::ctx::MAX_CASCADE;
use crate::compressor::ROOT_SCHEME_ID;

/// Most structural steps a [`SitePath`] holds. A deeper site has no path and is never pinned.
pub const MAX_SITE_DEPTH: usize = 8;

/// Structural step markers for a [`SitePath`]. Struct fields and union children use their index,
/// which must be below [`FIRST_MARKER`](site::FIRST_MARKER).
pub mod site {
    /// The elements of a list, list view or fixed-size list.
    pub const LIST_ELEMENTS: u32 = u32::MAX;
    /// The storage of an extension array.
    pub const EXTENSION_STORAGE: u32 = u32::MAX - 1;
    /// The type ids of a union array.
    pub const UNION_TYPE_IDS: u32 = u32::MAX - 2;
    /// The core storage of a variant array; its slots follow by index.
    pub const VARIANT_CORE: u32 = u32::MAX - 3;
    /// The shredded values of a variant array.
    pub const VARIANT_SHREDDED: u32 = u32::MAX - 4;
    /// The lowest marker; an index at or above it does not fit a path.
    pub const FIRST_MARKER: u32 = u32::MAX - 63;
}

/// Overflow marker for [`SitePath::len`].
const OVERFLOWED: u8 = u8::MAX;

/// The structural steps from a top-level array to a compression site: struct field and union
/// child indices, and the [`site`] markers, at most [`MAX_SITE_DEPTH`].
///
/// Carried by value in [`CompressorContext`](super::CompressorContext). Unused steps stay zero,
/// so derived equality and order compare only the path.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SitePath {
    len: u8,
    steps: [u32; MAX_SITE_DEPTH],
}

impl SitePath {
    /// The top-level array's own site.
    pub const ROOT: Self = Self {
        len: 0,
        steps: [0; MAX_SITE_DEPTH],
    };

    /// A path from its steps, or `None` if there are more than [`MAX_SITE_DEPTH`].
    pub fn new(steps: &[u32]) -> Option<Self> {
        let mut path = Self::ROOT;
        for &step in steps {
            path = path.child(step);
        }
        (!path.is_overflowed()).then_some(path)
    }

    /// This path extended by `step`. A path that is already full overflows.
    pub fn child(self, step: u32) -> Self {
        if self.is_overflowed() || usize::from(self.len) == MAX_SITE_DEPTH {
            return Self {
                len: OVERFLOWED,
                steps: [0; MAX_SITE_DEPTH],
            };
        }
        let mut steps = self.steps;
        steps[usize::from(self.len)] = step;
        Self {
            len: self.len + 1,
            steps,
        }
    }

    /// This path extended by a struct field or union child index.
    pub fn child_index(self, index: usize) -> Self {
        match u32::try_from(index) {
            Ok(step) if step < site::FIRST_MARKER => self.child(step),
            _ => Self {
                len: OVERFLOWED,
                steps: [0; MAX_SITE_DEPTH],
            },
        }
    }

    /// Whether the site is deeper than [`MAX_SITE_DEPTH`] (or below an index that does not fit).
    pub fn is_overflowed(&self) -> bool {
        self.len == OVERFLOWED
    }

    /// The steps, or `None` for an overflowed path.
    pub fn steps(&self) -> Option<&[u32]> {
        (!self.is_overflowed()).then(|| &self.steps[..usize::from(self.len)])
    }
}

impl fmt::Display for SitePath {
    /// Steps joined by `/`, markers by name (`elements`, `storage`, ...); the root is empty.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(steps) = self.steps() else {
            return f.write_str("overflowed");
        };
        for (i, step) in steps.iter().enumerate() {
            if i > 0 {
                f.write_str("/")?;
            }
            match *step {
                site::LIST_ELEMENTS => f.write_str("elements")?,
                site::EXTENSION_STORAGE => f.write_str("storage")?,
                site::UNION_TYPE_IDS => f.write_str("type_ids")?,
                site::VARIANT_CORE => f.write_str("variant_core")?,
                site::VARIANT_SHREDDED => f.write_str("variant_shredded")?,
                other => write!(f, "{other}")?,
            }
        }
        Ok(())
    }
}

impl fmt::Debug for SitePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SitePath({self})")
    }
}

/// A site within one top-level array: its structural path and the cascade history above it.
///
/// Fixed-size, so a sorted table of keys is searched without allocation.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SiteKey {
    path: SitePath,
    history_len: u8,
    /// `(scheme name, child index)` pairs from the top; unused entries stay `("", 0)`.
    history: [(&'static str, u32); MAX_CASCADE],
}

impl SiteKey {
    /// The key of a site, or `None` if its path overflowed or its history is deeper than
    /// [`MAX_CASCADE`].
    pub fn new(path: SitePath, history: &[(SchemeId, usize)]) -> Option<Self> {
        if path.is_overflowed() || history.len() > MAX_CASCADE {
            return None;
        }
        let mut entries = [("", 0); MAX_CASCADE];
        for (entry, (id, child)) in entries.iter_mut().zip(history) {
            *entry = (id.name, u32::try_from(*child).ok()?);
        }
        Some(Self {
            path,
            history_len: u8::try_from(history.len()).ok()?,
            history: entries,
        })
    }

    /// The structural path.
    pub fn path(&self) -> SitePath {
        self.path
    }

    /// The cascade history as `(scheme name, child index)` pairs from the top.
    pub fn history(&self) -> &[(&'static str, u32)] {
        &self.history[..usize::from(self.history_len)]
    }
}

impl fmt::Display for SiteKey {
    /// `path | history`, the history as `scheme[child]` joined by ` > `, the compressor's own
    /// list cascading as `vortex.compressor.root[child]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} | ", self.path)?;
        for (i, (name, child)) in self.history().iter().enumerate() {
            if i > 0 {
                f.write_str(" > ")?;
            }
            write!(f, "{name}[{child}]")?;
        }
        Ok(())
    }
}

impl fmt::Debug for SiteKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SiteKey({self})")
    }
}

/// The name [`SiteKey`] histories use for the compressor's own list cascading.
pub const ROOT_SCHEME_NAME: &str = ROOT_SCHEME_ID.name;
