"""Build the fused table+candidates+ontology heterogeneous graph.

Implements the contract in docs/SCHEMA.md. Gate 2 (CLAUDE.md s3) tests this.

The graph is the unit of batching: each table is an independent PyG HeteroData,
mini-batched with PyG's loader. Node/edge typing follows SCHEMA.md exactly; the
two ontology-aware edge attributes (`subclass_of.hop`, `kg_triple.property_id`)
are what the ontology-conditioned message function reads (MODEL_SPEC.md s2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import torch

from flint.data.kg import KG, ancestors_with_hops, literal_matches
from flint.graph.features import TextEncoder

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData


@dataclass
class TableInput:
    """A single relational table plus the inputs needed to build its graph."""

    table_id: str
    headers: Sequence[str]
    rows: Sequence[Sequence[str]]
    # candidates[col][row] -> list of candidate entity ids for that cell
    candidates: Sequence[Sequence[Sequence[str]]]
    entity_columns: Sequence[int]  # indices of columns we attempt to link


@dataclass
class OntologyNeighborhood:
    """The slice of Wikidata relevant to one table.

    Extracted lazily per-table to keep graphs small (Wikidata has millions of
    classes; we only attach the local neighborhood of the candidates' types).
    `depth[class]` = min subclass-of hops from any candidate direct type to that
    class (0 for a direct type); it drives the subclass_of `hop` edge attribute.
    """

    classes: set[str] = field(default_factory=set)
    properties: set[str] = field(default_factory=set)
    subclass_edges: list[tuple[str, str, int]] = field(default_factory=list)  # (child, parent, hop)
    class_properties: list[tuple[str, str]] = field(default_factory=list)     # (class, property)
    depth: dict[str, int] = field(default_factory=dict)


def extract_ontology_neighborhood(
    candidate_type_ids: set[str],
    *,
    max_hops: int,
    kg_index: KG,
) -> OntologyNeighborhood:
    """Walk up the subclass-of hierarchy from candidate types up to `max_hops`.

    Returns the connected ontology neighborhood to attach to the graph. Edges go
    child -> parent and are kept only when they increase depth, guaranteeing a DAG
    (SCHEMA invariant 4). `max_hops` generalizes GRAMS+'s hand-set max_distance=2:
    we attach the structure and let the model LEARN how far to trust ancestors.
    """
    depth: dict[str, int] = {}
    for t in candidate_type_ids:
        for cls, h in ancestors_with_hops(kg_index, t, max_hops=max_hops).items():
            if cls not in depth or h < depth[cls]:
                depth[cls] = h

    classes = set(depth)
    subclass_edges: list[tuple[str, str, int]] = []
    for child in classes:
        for parent in kg_index.subclass_of(child):
            if parent in classes and depth[parent] > depth[child]:
                subclass_edges.append((child, parent, depth[parent]))

    return OntologyNeighborhood(
        classes=classes,
        subclass_edges=subclass_edges,
        depth=depth,
    )


def build_hetero_graph(
    table: TableInput,
    ontology: OntologyNeighborhood,
    *,
    feature_encoder: TextEncoder,
    kg: KG,
    prop_vocab: dict[str, int] | None = None,
    max_candidate_types: int = 8,
) -> "HeteroData":
    """Compile one table into a PyG HeteroData per docs/SCHEMA.md.

    Args:
        table: the table + per-cell candidate entity ids.
        ontology: the class hierarchy neighborhood (extract_ontology_neighborhood).
        feature_encoder: text encoder for header / label / description features.
        kg: ontology access for candidate instance_of and statements.
        prop_vocab: optional shared {property_id: index} map for the kg_triple
            edge attribute (and the CPA head's E_prop). Grown in place if given.
        max_candidate_types: cap on instance_of types attached per candidate.

    Returns:
        HeteroData satisfying every invariant in docs/SCHEMA.md. Node id maps are
        attached (e.g. data['cell_candidate'].entity_ids) for inspection/eval.
    """
    from torch_geometric.data import HeteroData

    enc = feature_encoder
    if prop_vocab is None:
        prop_vocab = {}
    data = HeteroData()

    # ----- column nodes (ALL columns; literals included for CPA head) ----------
    ncols = len(table.headers)
    entity_cols = set(table.entity_columns)
    col_headers = [str(h) for h in table.headers]
    header_emb = enc.encode(col_headers)
    is_entity = torch.tensor([[1.0 if c in entity_cols else 0.0] for c in range(ncols)])
    position = torch.tensor([[c / max(ncols - 1, 1)] for c in range(ncols)])
    data["column"].x = torch.cat([header_emb, is_entity, position], dim=1)
    data["column"].col_index = list(range(ncols))

    # ----- cell_candidate nodes ------------------------------------------------
    cand_entity: list[str] = []
    cand_row: list[int] = []
    cand_col: list[int] = []
    # candidates[col] is indexed by position within entity_columns order
    for ci, col in enumerate(table.entity_columns):
        col_cands = table.candidates[ci] if ci < len(table.candidates) else []
        for row, cell_cands in enumerate(col_cands):
            for ent in cell_cands:
                cand_entity.append(ent)
                cand_row.append(row)
                cand_col.append(col)
    ncand = len(cand_entity)
    cand_labels = [kg.label(e) for e in cand_entity]
    data["cell_candidate"].x = (
        enc.encode(cand_labels) if ncand else torch.zeros((0, enc.dim))
    )
    data["cell_candidate"].entity_ids = cand_entity
    data["cell_candidate"].row = torch.tensor(cand_row, dtype=torch.long)
    data["cell_candidate"].col = torch.tensor(cand_col, dtype=torch.long)

    # map entity_id -> candidate node indices (an entity can occur in many cells)
    ent_to_nodes: dict[str, list[int]] = {}
    for idx, e in enumerate(cand_entity):
        ent_to_nodes.setdefault(e, []).append(idx)

    # ----- onto_class nodes ----------------------------------------------------
    class_ids = sorted(ontology.classes)
    class_idx = {c: i for i, c in enumerate(class_ids)}
    class_labels = [kg.label(c) for c in class_ids]
    class_depth = torch.tensor([[float(ontology.depth.get(c, 0))] for c in class_ids])
    data["onto_class"].x = (
        torch.cat([enc.encode(class_labels), class_depth], dim=1)
        if class_ids
        else torch.zeros((0, enc.dim + 1))
    )
    data["onto_class"].class_ids = class_ids

    # ----- instance_of edges (candidate -> onto_class) + collect properties ----
    iof_src, iof_dst = [], []
    class_prop_pairs: set[tuple[str, str]] = set()
    kgt_src, kgt_dst, kgt_prop = [], [], []
    for idx, ent in enumerate(cand_entity):
        types = [t for t in kg.instance_of(ent)[:max_candidate_types] if t in class_idx]
        for t in types:
            iof_src.append(idx)
            iof_dst.append(class_idx[t])
        # statements -> kg_triple edges to other candidates + has_property linkage
        for pid, obj in kg.statements(ent):
            tgt_nodes = ent_to_nodes.get(obj, [])
            if not tgt_nodes:
                continue
            pidx = prop_vocab.setdefault(pid, len(prop_vocab))
            for j in tgt_nodes:
                if j == idx:
                    continue
                kgt_src.append(idx)
                kgt_dst.append(j)
                kgt_prop.append(pidx)
                for t in types:
                    class_prop_pairs.add((t, pid))

    # ----- literal_match edges (cell_candidate -> column, property_id attr) -----
    # entity->literal CPA evidence: a candidate's literal statement value matches a
    # literal column's cell in the same row (e.g. P1351 points = the Points cell).
    entity_col_set = set(table.entity_columns)
    literal_cols = [c for c in range(ncols) if c not in entity_col_set]
    lm_src, lm_dst, lm_prop = [], [], []
    if literal_cols:
        for idx, ent in enumerate(cand_entity):
            r = cand_row[idx]
            lits = kg.literal_statements(ent)
            if not lits:
                continue
            for c_lit in literal_cols:
                cell = table.rows[r][c_lit] if r < len(table.rows) else ""
                for pid, val, kind in lits:
                    if literal_matches(cell, val, kind):
                        pidx = prop_vocab.setdefault(pid, len(prop_vocab))
                        lm_src.append(idx)
                        lm_dst.append(c_lit)
                        lm_prop.append(pidx)
                        class_prop_pairs.update((t, pid) for t in kg.instance_of(ent)[:max_candidate_types] if t in class_idx)
                        break  # one matching property per (candidate, literal col)

    # ----- onto_property nodes (observed statement properties) -----------------
    prop_ids = sorted({p for _, p in class_prop_pairs} | set(ontology.properties))
    prop_local = {p: i for i, p in enumerate(prop_ids)}
    prop_labels = [kg.label(p) for p in prop_ids]
    data["onto_property"].x = (
        enc.encode(prop_labels) if prop_ids else torch.zeros((0, enc.dim))
    )
    data["onto_property"].prop_ids = prop_ids

    # ----- has_property edges (onto_class -> onto_property) ---------------------
    hp_src, hp_dst = [], []
    for c, p in sorted(class_prop_pairs):
        if c in class_idx and p in prop_local:
            hp_src.append(class_idx[c])
            hp_dst.append(prop_local[p])

    # ----- in_column edges (cell_candidate -> column) --------------------------
    inc_src = list(range(ncand))
    inc_dst = cand_col  # exactly one per candidate (SCHEMA invariant 2)

    # ----- same_row edges (within a row, across entity columns) ----------------
    by_row: dict[int, list[int]] = {}
    for idx, r in enumerate(cand_row):
        by_row.setdefault(r, []).append(idx)
    sr_src, sr_dst = [], []
    for nodes in by_row.values():
        for a in nodes:
            for b in nodes:
                if a != b:
                    sr_src.append(a)
                    sr_dst.append(b)

    # ----- subclass_of edges (onto_class -> onto_class, hop attr) ---------------
    sc_src, sc_dst, sc_hop = [], [], []
    for child, parent, hop in ontology.subclass_edges:
        if child in class_idx and parent in class_idx:
            sc_src.append(class_idx[child])
            sc_dst.append(class_idx[parent])
            sc_hop.append(hop)

    def _ei(src: list[int], dst: list[int]) -> torch.Tensor:
        if not src:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor([src, dst], dtype=torch.long)

    data["cell_candidate", "in_column", "column"].edge_index = _ei(inc_src, inc_dst)
    data["cell_candidate", "same_row", "cell_candidate"].edge_index = _ei(sr_src, sr_dst)
    data["cell_candidate", "instance_of", "onto_class"].edge_index = _ei(iof_src, iof_dst)
    sc = data["onto_class", "subclass_of", "onto_class"]
    sc.edge_index = _ei(sc_src, sc_dst)
    sc.edge_attr = torch.tensor(sc_hop, dtype=torch.long).view(-1, 1)
    data["onto_class", "has_property", "onto_property"].edge_index = _ei(hp_src, hp_dst)
    kt = data["cell_candidate", "kg_triple", "cell_candidate"]
    kt.edge_index = _ei(kgt_src, kgt_dst)
    kt.edge_attr = torch.tensor(kgt_prop, dtype=torch.long)
    lm = data["cell_candidate", "literal_match", "column"]
    lm.edge_index = _ei(lm_src, lm_dst)
    lm.edge_attr = torch.tensor(lm_prop, dtype=torch.long)

    data.table_id = table.table_id
    # expose the property vocab so the model can map kg_triple indices -> P-ids
    data.prop_vocab = dict(prop_vocab)
    return data


# --------------------------------------------------------------------------- #
# Convenience: STIRecord -> graph (250WT end-to-end, candidates seeded from CEA)
# --------------------------------------------------------------------------- #
def build_graph_from_record(
    record,  # noqa: ANN001  (flint.data.loaders.STIRecord)
    *,
    kg: KG,
    feature_encoder: TextEncoder,
    max_hops: int = 3,
    prop_vocab: dict[str, int] | None = None,
) -> "HeteroData":
    """Build a table graph from a loaded STIRecord.

    For the fallback baseline (CLAUDE.md s4) we seed each entity cell's candidate
    set from the gold CEA link (GRAMS-style "given the entity"); richer multi-
    candidate retrieval (DATA_SPEC Part A) is a later refinement.
    """
    # candidates[ci][row] for each entity column, from gold CEA
    candidates: list[list[list[str]]] = []
    for col in record.entity_columns:
        col_cands: list[list[str]] = []
        for row in range(len(record.rows)):
            ent = record.gold_cea.get((row, col))
            col_cands.append([ent] if ent else [])
        candidates.append(col_cands)

    cand_type_ids: set[str] = set()
    for col_cands in candidates:
        for cell in col_cands:
            for ent in cell:
                cand_type_ids.update(kg.instance_of(ent))

    ontology = extract_ontology_neighborhood(cand_type_ids, max_hops=max_hops, kg_index=kg)
    table = TableInput(
        table_id=record.table_id,
        headers=record.headers,
        rows=record.rows,
        candidates=candidates,
        entity_columns=record.entity_columns,
    )
    return build_hetero_graph(
        table, ontology, feature_encoder=feature_encoder, kg=kg, prop_vocab=prop_vocab
    )
