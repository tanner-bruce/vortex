// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ZY-615 phase 0 measurement only, never merged: seeds a [`SchemePlan`] from the encoding trees
//! of the input parts' chunks, read from their layouts without decoding data.
//!
//! Each chunk's tree votes, weighted by the chunk's rows, for one decision at every compression
//! site it shows: the scheme that produced the node there, or canonical for an uncompressed
//! node. A constant node votes for nothing (the compressor encodes constant, empty and all-null
//! arrays itself before any plan lookup, masking a constant with nulls), and neither do the
//! elements of lists whose offsets are constant, which are all empty. Validity and patches are not compression sites.
//!
//! The plan takes the decision with the most rows at each site, top down: a child site counts
//! only when its cascade history follows the decisions taken above it. A node this module does
//! not know votes nowhere below it and is reported.
//!
//! ZY-615 cells 4: [`PlanVotes::seed_races`] makes every voted site a race between its top two
//! decisions, with Zstd added at string sites, and [`seed_described`] seeds one decision from a
//! line of [`SchemePlan::describe`], as a pin list written offline holds them.

use std::collections::BTreeMap;
use std::collections::HashMap;

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::SerializedArray;
use vortex_compressor::SchemePlan;
use vortex_compressor::plan_site;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::scheme::SchemeExt;
use vortex_compressor::scheme::SchemeId;

use crate::ALL_SCHEMES;
use crate::schemes::temporal::TemporalScheme;

/// The id of the scheme named `name`, among every scheme this build has, or the compressor's
/// root cascade step's.
pub fn scheme_id(name: &str) -> Option<SchemeId> {
    let root = SchemePlan::list_offsets_step().0;
    if name == root.to_string() {
        return Some(root);
    }
    let gated: &[&dyn Scheme] = &[
        #[cfg(feature = "zstd")]
        &crate::schemes::string::ZstdScheme,
        #[cfg(feature = "zstd")]
        &crate::schemes::string::ZstdBuffersScheme,
        #[cfg(feature = "zstd")]
        &crate::schemes::binary::ZstdScheme,
        #[cfg(feature = "zstd")]
        &crate::schemes::binary::ZstdBuffersScheme,
        #[cfg(feature = "pco")]
        &crate::schemes::integer::PcoScheme,
        #[cfg(feature = "pco")]
        &crate::schemes::float::PcoScheme,
    ];
    ALL_SCHEMES
        .iter()
        .chain(gated)
        .find(|scheme| scheme.scheme_name() == name)
        .map(|scheme| scheme.id())
}

type Site = (Vec<u32>, Vec<(SchemeId, usize)>);

/// Row-weighted votes per compression site from input chunks' encoding trees.
#[derive(Debug, Default)]
pub struct PlanVotes {
    votes: HashMap<Site, HashMap<Option<SchemeId>, u64>>,
    /// Voted sites whose arrays are strings (`true`) or binary (`false`).
    strings: HashMap<Site, bool>,
    unknown: BTreeMap<String, u64>,
    chunks: u64,
    rows: u64,
}

/// One seeded site: its path, the decision taken, and its rows over all rows voting there.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct SeededSite {
    pub site: String,
    pub decision: String,
    pub rows: u64,
    pub voting_rows: u64,
    pub alternatives: usize,
}

fn int() -> DType {
    DType::Primitive(PType::U64, Nullability::NonNullable)
}

impl PlanVotes {
    /// Empty votes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Chunks and rows added.
    pub fn totals(&self) -> (u64, u64) {
        (self.chunks, self.rows)
    }

    /// Nodes no decision could be read from, as `encoding under dtype`, with their rows.
    pub fn unknown(&self) -> &BTreeMap<String, u64> {
        &self.unknown
    }

    /// Adds one chunk of `rows` rows of a column of `dtype`, whose array tree is `tree` and whose
    /// encoding ids resolve through `names`.
    pub fn add_chunk(&mut self, dtype: &DType, tree: &SerializedArray, names: &[String], rows: u64) {
        self.chunks += 1;
        self.rows += rows;
        let mut walk = Walk {
            votes: self,
            names,
            rows,
        };
        walk.node(dtype, tree, &mut Vec::new(), &mut Vec::new());
    }

    /// Makes every voted site of `plan` a race between its two decisions with the most rows,
    /// with string or binary Zstd added at a string or binary site when `zstd`. A site `plan`
    /// already has a decision for keeps it. Returns the race sites made.
    pub fn seed_races(&self, plan: &SchemePlan, zstd: bool) -> usize {
        let label = |decision: &Option<SchemeId>| decision.map(|id| id.to_string());
        let mut made = 0;
        for (key, votes) in &self.votes {
            let (site, history) = key;
            if plan.is_seeded(site, history) {
                continue;
            }
            let mut ranked = votes.iter().collect::<Vec<_>>();
            ranked.sort_by(|a, b| b.1.cmp(a.1).then_with(|| label(a.0).cmp(&label(b.0))));
            let top = ranked.iter().take(2).map(|(decision, _)| **decision).collect::<Vec<_>>();
            let extra = match self.strings.get(key) {
                Some(true) if zstd => scheme_id("vortex.string.zstd"),
                Some(false) if zstd => scheme_id("vortex.binary.zstd"),
                _ => None,
            };
            plan.race(site, history, &top, extra);
            made += 1;
        }
        made
    }

    /// Seeds `plan` with the decision with the most rows at each reachable site and returns the
    /// seeded sites in site order.
    pub fn seed(&self, plan: &SchemePlan) -> Vec<SeededSite> {
        let mut keys = self.votes.keys().collect::<Vec<_>>();
        keys.sort_by(|a, b| (&a.0, a.1.len(), describe(a)).cmp(&(&b.0, b.1.len(), describe(b))));
        let mut chosen = HashMap::<Site, Option<SchemeId>>::new();
        let mut seeded = Vec::new();
        let root = SchemePlan::list_offsets_step().0;
        for key in keys {
            let (site, history) = key;
            let reachable = (0..history.len()).all(|depth| {
                let (scheme, _) = history[depth];
                scheme == root
                    || chosen.get(&(site.clone(), history[..depth].to_vec())) == Some(&Some(scheme))
            });
            if !reachable {
                continue;
            }
            let votes = &self.votes[key];
            let voting_rows = votes.values().sum();
            // Most rows, then the first name (canonical first), so ties are stable.
            let label = |decision: &Option<SchemeId>| decision.map(|id| id.to_string());
            let (decision, rows) = votes
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| label(b.0).cmp(&label(a.0))))
                .map(|(decision, rows)| (*decision, *rows))
                .unwrap_or((None, 0));
            plan.seed(site, history, decision);
            chosen.insert(key.clone(), decision);
            seeded.push(SeededSite {
                site: describe(key),
                decision: decision.map_or_else(|| "canonical".to_owned(), |id| id.to_string()),
                rows,
                voting_rows,
                alternatives: votes.len(),
            });
        }
        seeded
    }
}

/// Seeds `plan` with one decision written as [`SchemePlan::describe`] writes it: `site` is the
/// site path (`0/elements`, empty at the column), `history` the cascade (`scheme[child] > ...`,
/// empty at a site's top) and `decision` a scheme name or `canonical`.
pub fn seed_described(
    plan: &SchemePlan,
    site: &str,
    history: &str,
    decision: &str,
) -> Result<(), String> {
    let path = site
        .split('/')
        .filter(|part| !part.is_empty())
        .map(|part| match part {
            "elements" => Ok(plan_site::LIST_ELEMENTS),
            "storage" => Ok(plan_site::EXTENSION_STORAGE),
            "type_ids" => Ok(plan_site::UNION_TYPE_IDS),
            index => index
                .parse::<u32>()
                .map_err(|_| format!("site {site:?}: part {index:?}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let steps = history
        .split(" > ")
        .filter(|step| !step.is_empty())
        .map(|step| {
            let (name, child) = step
                .strip_suffix(']')
                .and_then(|step| step.rsplit_once('['))
                .ok_or_else(|| format!("history {history:?}: step {step:?}"))?;
            let id = scheme_id(name).ok_or_else(|| format!("history {history:?}: scheme {name:?}"))?;
            let child = child
                .parse::<usize>()
                .map_err(|_| format!("history {history:?}: child {child:?}"))?;
            Ok((id, child))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let decision = match decision {
        "canonical" => None,
        name => Some(scheme_id(name).ok_or_else(|| format!("decision {name:?}"))?),
    };
    plan.seed(&path, &steps, decision);
    Ok(())
}

fn describe((site, history): &Site) -> String {
    let mut line = String::new();
    for (i, index) in site.iter().enumerate() {
        if i > 0 {
            line.push('/');
        }
        match *index {
            plan_site::LIST_ELEMENTS => line.push_str("elements"),
            plan_site::EXTENSION_STORAGE => line.push_str("storage"),
            other => line.push_str(&other.to_string()),
        }
    }
    line.push_str(" |");
    for (i, (id, child)) in history.iter().enumerate() {
        line.push_str(if i > 0 { " > " } else { " " });
        line.push_str(&format!("{id}[{child}]"));
    }
    line
}

struct Walk<'a> {
    votes: &'a mut PlanVotes,
    names: &'a [String],
    rows: u64,
}

impl Walk<'_> {
    fn vote(&mut self, site: &[u32], history: &[(SchemeId, usize)], decision: Option<SchemeId>) {
        *self
            .votes
            .votes
            .entry((site.to_vec(), history.to_vec()))
            .or_default()
            .entry(decision)
            .or_default() += self.rows;
    }

    fn unknown(&mut self, name: &str, dtype: &DType) {
        *self
            .votes
            .unknown
            .entry(format!("{name} under {dtype}"))
            .or_default() += self.rows;
    }

    fn name(&self, node: &SerializedArray) -> &str {
        self.names
            .get(usize::from(node.encoding_id()))
            .map_or("?", String::as_str)
    }

    /// Votes at `site`/`history` for `node`, a chunk of `dtype` there, and below it.
    fn node(
        &mut self,
        dtype: &DType,
        node: &SerializedArray,
        site: &mut Vec<u32>,
        history: &mut Vec<(SchemeId, usize)>,
    ) {
        let name = self.name(node).to_owned();
        // The compressor's own constant encoding, masked when the constant has nulls.
        if name == "vortex.constant"
            || name == "vortex.null"
            || (name == "vortex.masked"
                && node.nchildren() >= 1
                && self.name(&node.child(0)) == "vortex.constant")
        {
            return;
        }
        match dtype {
            DType::Struct(fields, _) if name == "vortex.struct" => {
                // A nullable struct's validity comes before its fields.
                let offset = node.nchildren().saturating_sub(fields.nfields());
                for (index, field) in fields.fields().enumerate() {
                    if offset + index >= node.nchildren() {
                        return self.unknown(&name, dtype);
                    }
                    self.structural(site, index as u32, |walk, site| {
                        walk.node(&field, &node.child(offset + index), site, &mut Vec::new());
                    });
                }
            }
            DType::List(element, _) if name == "vortex.list" || name == "vortex.listview" => {
                if node.nchildren() < 2 {
                    return self.unknown(&name, dtype);
                }
                let offsets = node.child(1);
                if self.name(&offsets) == "vortex.constant" {
                    // Every list is empty: nothing below votes.
                    return;
                }
                self.structural(site, plan_site::LIST_ELEMENTS, |walk, site| {
                    walk.node(element, &node.child(0), site, &mut Vec::new());
                });
                self.child(&int(), &offsets, site, history, SchemePlan::list_offsets_step());
            }
            DType::FixedSizeList(element, ..) if name == "vortex.fixed_size_list" => {
                if node.nchildren() < 1 {
                    return self.unknown(&name, dtype);
                }
                self.structural(site, plan_site::LIST_ELEMENTS, |walk, site| {
                    walk.node(element, &node.child(0), site, &mut Vec::new());
                });
            }
            DType::Extension(ext) => {
                let storage = ext.storage_dtype().clone();
                match name.as_str() {
                    // The storage compression won at this chunk: no scheme on the extension.
                    "vortex.ext" if node.nchildren() >= 1 => {
                        self.vote(site, history, None);
                        self.structural(site, plan_site::EXTENSION_STORAGE, |walk, site| {
                            walk.node(&storage, &node.child(0), site, &mut Vec::new());
                        });
                    }
                    // The temporal scheme won: its parts cascade, and the storage stays canonical.
                    "vortex.datetimeparts" if node.nchildren() >= 3 => {
                        let id = TemporalScheme.id();
                        self.vote(site, history, Some(id));
                        for part in 0..3 {
                            self.child(&int(), &node.child(part), site, history, (id, part));
                        }
                        self.structural(site, plan_site::EXTENSION_STORAGE, |walk, site| {
                            walk.vote(site, &[], None);
                        });
                    }
                    _ => self.unknown(&name, dtype),
                }
            }
            _ => self.leaf(dtype, &name, node, site, history),
        }
    }

    /// Runs `f` at the structural child `index` of `site`, which starts a fresh cascade.
    fn structural(
        &mut self,
        site: &mut Vec<u32>,
        index: u32,
        f: impl FnOnce(&mut Self, &mut Vec<u32>),
    ) {
        site.push(index);
        f(self, site);
        site.pop();
    }

    /// Votes for `node`, the cascaded child `step` of the node at `site`/`history`.
    fn child(
        &mut self,
        dtype: &DType,
        node: &SerializedArray,
        site: &mut Vec<u32>,
        history: &mut Vec<(SchemeId, usize)>,
        step: (SchemeId, usize),
    ) {
        history.push(step);
        self.node(dtype, node, site, history);
        history.pop();
    }

    /// A non-structural node: the scheme that made it, by encoding and dtype, and the children
    /// that scheme cascaded, as (array child position, scheme child index, dtype).
    fn leaf(
        &mut self,
        dtype: &DType,
        name: &str,
        node: &SerializedArray,
        site: &mut Vec<u32>,
        history: &mut Vec<(SchemeId, usize)>,
    ) {
        let same = dtype.clone();
        let codes_values = || vec![(0, 1, int()), (1, 0, same.clone())];
        let (scheme, children): (Option<&str>, Vec<(usize, usize, DType)>) = match (dtype, name) {
            (DType::Primitive(..), "vortex.primitive")
            | (DType::Bool(_), "vortex.bool")
            | (DType::Utf8(_) | DType::Binary(_), "vortex.varbinview")
            | (DType::Decimal(..), "vortex.decimal") => (None, vec![]),
            (DType::Primitive(p, _), _) if p.is_int() => match name {
                "fastlanes.bitpacked" => (Some("vortex.int.bitpacking"), vec![]),
                "fastlanes.for" => (Some("vortex.int.for"), vec![]),
                "vortex.sequence" => (Some("vortex.int.sequence"), vec![]),
                "vortex.pco" => (Some("vortex.int.pco"), vec![]),
                "vortex.dict" => (Some("vortex.int.dict"), codes_values()),
                "vortex.runend" => (Some("vortex.int.runend"), codes_values()),
                "vortex.sparse" => (Some("vortex.int.sparse"), codes_values()),
                "vortex.zigzag" => (Some("vortex.int.zigzag"), vec![(0, 0, int())]),
                "fastlanes.delta" => (
                    Some("vortex.int.delta"),
                    vec![(0, 0, int()), (1, 1, int())],
                ),
                "fastlanes.rle" => (
                    Some("vortex.int.rle"),
                    vec![(0, 0, same.clone()), (1, 1, int()), (2, 2, int())],
                ),
                _ => (Some(""), vec![]),
            },
            (DType::Primitive(..), _) => match name {
                "vortex.alp" => (Some("vortex.float.alp"), vec![(0, 0, int())]),
                "vortex.alprd" => (Some("vortex.float.alprd"), vec![]),
                "vortex.pco" => (Some("vortex.float.pco"), vec![]),
                "vortex.dict" => (Some("vortex.float.dict"), codes_values()),
                "vortex.sparse" => (Some("vortex.float.sparse"), vec![(0, 0, int())]),
                "fastlanes.rle" => (
                    Some("vortex.float.rle"),
                    vec![(0, 0, same.clone()), (1, 1, int()), (2, 2, int())],
                ),
                _ => (Some(""), vec![]),
            },
            (DType::Utf8(_), _) => match name {
                "vortex.dict" => (Some("vortex.string.dict"), codes_values()),
                "vortex.fsst" => (
                    Some("vortex.string.fsst"),
                    vec![(0, 0, int()), (1, 1, int())],
                ),
                "vortex.onpair" => (
                    Some("vortex.string.onpair"),
                    vec![(0, 0, int()), (1, 1, int()), (2, 2, int()), (3, 3, int())],
                ),
                "vortex.zstd" => (Some("vortex.string.zstd"), vec![]),
                "vortex.zstd_buffers" => (Some("vortex.string.zstd_buffers"), vec![]),
                "vortex.sparse" => (Some("vortex.string.sparse"), vec![(0, 0, int())]),
                _ => (Some(""), vec![]),
            },
            (DType::Binary(_), _) => match name {
                "vortex.dict" => (Some("vortex.binary.dict"), codes_values()),
                "vortex.zstd" => (Some("vortex.binary.zstd"), vec![]),
                "vortex.zstd_buffers" => (Some("vortex.binary.zstd_buffers"), vec![]),
                "vortex.varbin" => (Some("vortex.binary.varbin"), vec![(0, 0, int())]),
                _ => (Some(""), vec![]),
            },
            (DType::Decimal(..), "vortex.decimal_byte_parts") => {
                (Some("vortex.decimal.byte_parts"), vec![(0, 0, int())])
            }
            _ => (Some(""), vec![]),
        };
        let decision = match scheme {
            None => None,
            Some(scheme) => match scheme_id(scheme) {
                Some(id) => Some(id),
                None => return self.unknown(name, dtype),
            },
        };
        if children.iter().any(|(position, ..)| *position >= node.nchildren()) {
            return self.unknown(name, dtype);
        }
        if let DType::Utf8(_) | DType::Binary(_) = dtype {
            self.votes
                .strings
                .insert((site.to_vec(), history.to_vec()), dtype.is_utf8());
        }
        self.vote(site, history, decision);
        if let Some(id) = decision {
            for (position, index, child_dtype) in children {
                self.child(&child_dtype, &node.child(position), site, history, (id, index));
            }
        }
    }
}
