"""Dataset loaders + leakage guard.

Each loader yields a uniform record so downstream code is dataset-agnostic:
    {table_id, headers, rows, entity_columns, gold_cta, gold_cpa, gold_cea}

The leakage guard (CLAUDE.md s2) MUST run for the distant-supervision train set.

Status: 250WT implemented (Gate 1) on top of Binh Vu's `sm` (sem-desc) loader, which
parses the canonical semantic-model ground truth. Other datasets are Gate-2+ work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from flint.utils.paths import dataset_dir

# Wikidata id (Qxxx / Pxxx) extracted from a full URI like
# http://www.wikidata.org/entity/Q5 or .../prop/P102 -> "Q5" / "P102".
_WD_ID = re.compile(r"(?:entity|prop(?:/[a-z]+)?|prop/direct)/([QP]\d+)$")
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_STATEMENT = "http://wikiba.se/ontology#Statement"


def _wd_id(uri: str) -> str | None:
    """Return the bare Q/P id from a Wikidata URI, or None if not a Q/P entity/prop."""
    m = _WD_ID.search(uri)
    if m:
        return m.group(1)
    # tolerate the plain ".../prop/P102" form not caught above
    tail = uri.rstrip("/").rsplit("/", 1)[-1]
    return tail if re.fullmatch(r"[QP]\d+", tail) else None


@dataclass
class STIRecord:
    table_id: str
    headers: list[str]
    rows: list[list[str]]
    entity_columns: list[int]
    gold_cta: dict[int, str]              # col_idx -> class id (Qxxx) (primary/most-specific)
    gold_cpa: dict[tuple[int, int], str]  # (subj_col, obj_col) -> property id (Pxxx)
    gold_cea: dict[tuple[int, int], str]  # (row, col) -> entity id (Qxxx)
    # provenance for CPA edges that were reified through a wikibase:Statement node,
    # so a later rigor pass can reconcile main-property vs qualifier conventions
    # against the GRAMS+ evaluator before any CPA number is reported.
    cpa_via_statement: dict[tuple[int, int], dict] = field(default_factory=dict)
    # multi-valid-type CTA gold (SemTab ToughTables/tFood): col -> SET of acceptable
    # classes (exact + ancestors). Eval gives full credit for ANY member. For 250WT
    # this is {col: {gold_cta[col]}}.
    gold_cta_accept: dict[int, set] = field(default_factory=dict)


def load_dataset(name: str, split: str = "all") -> Iterator[STIRecord]:
    """name in {250wt, hardtables, wikidatatables, toughtables, tfood}."""
    name = name.lower().replace("-", "").replace("_", "")
    if name in {"250wt", "wt250"}:
        yield from _load_250wt()
        return
    if name in {"toughtables", "2twd", "2t"}:
        yield from _load_toughtables()
        return
    if name in {"hardtables", "hardtablesr1"}:
        yield from _load_semtab("hardtables", split)
        return
    if name in {"wikidatatables", "wikidatatables2023r1"}:
        yield from _load_semtab("wikidatatables", split)
        return
    if name in {"tfood"}:
        yield from _load_semtab("tfood", split)
        return
    raise NotImplementedError(
        f"loader for {name!r} not implemented; have 250wt, toughtables, hardtables, wikidatatables")


def _load_semtab(name: str, split: str = "all") -> Iterator[STIRecord]:
    """Generic SemTab-format loader (HardTables R1, WikidataTables 2023 R1).

    Layout (auto-discovered): .../<Split>/{tables/<tid>.csv, gt/{cta,cpa,cea}_gt.csv}.
    gt rows: cta=(tid,col,type_url); cpa=(tid,subj,obj,prop_url); cea=(tid,row,col,ent_url).
    Tables have generic headers (col0,col1,...). gold_cea provides oracle entities so the
    identical-protocol (gold-entity) eval matches the 250WT headline. `split` in
    {valid, test, all}; default picks every split that ships non-blind gt.
    """
    import csv

    base = dataset_dir(name)
    want = split.lower()
    # discover split dirs by their cta_gt.csv (split dir = .../<Split>/, has tables/ + gt/)
    split_dirs = []
    for ctp in sorted(base.glob("**/gt/cta_gt.csv")):
        sd = ctp.parent.parent
        if not (sd / "tables").is_dir():
            continue
        sname = sd.name.lower()
        if want != "all" and want not in sname:
            continue
        if ctp.stat().st_size == 0:
            continue
        split_dirs.append(sd)

    seen: set[str] = set()
    for sd in split_dirs:
        gt = sd / "gt"
        cta: dict[str, dict[int, str]] = {}
        with open(gt / "cta_gt.csv", newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 3 and (q := _wd_id(row[2])):
                    cta.setdefault(row[0], {})[int(row[1])] = q
        cpa: dict[str, dict[tuple[int, int], str]] = {}
        cpf = gt / "cpa_gt.csv"
        if cpf.exists():
            with open(cpf, newline="") as f:
                for row in csv.reader(f):
                    if len(row) >= 4 and (p := _wd_id(row[3])):
                        cpa.setdefault(row[0], {})[(int(row[1]), int(row[2]))] = p
        cea: dict[str, dict[tuple[int, int], str]] = {}
        cef = gt / "cea_gt.csv"
        if cef.exists():
            with open(cef, newline="") as f:
                for row in csv.reader(f):
                    if len(row) >= 4 and (e := _wd_id(row[3])):
                        # SemTab CEA row index is 1-based incl. header (row 0); we strip
                        # the header into rows[0], so subtract 1 to align entity<->cell row.
                        # (off-by-one here silently broke CPA: subject entity paired with
                        #  the wrong row's object cell -> candidate value-match failed.)
                        cea.setdefault(row[0], {})[(int(row[1]) - 1, int(row[2]))] = e
        for tid in sorted(cta):
            if tid in seen:
                continue
            fp = sd / "tables" / f"{tid}.csv"
            if not fp.exists():
                continue
            seen.add(tid)
            with open(fp, newline="") as f:
                rows_all = list(csv.reader(f))
            if not rows_all:
                continue
            headers = rows_all[0]
            rows = rows_all[1:]
            ncols = len(headers)
            gold_cta = {c: q for c, q in cta[tid].items() if c < ncols}
            gold_cea = {k: e for k, e in cea.get(tid, {}).items() if k[1] < ncols}
            ent_cols = sorted(set(gold_cta) | {c for (_, c) in gold_cea})
            yield STIRecord(
                table_id=tid,
                headers=[str(h) for h in headers],
                rows=[[str(x) for x in r] + [""] * (ncols - len(r)) for r in rows],
                entity_columns=ent_cols,
                gold_cta=gold_cta,
                gold_cpa=cpa.get(tid, {}),
                gold_cea=gold_cea,
            )


def _load_toughtables() -> Iterator[STIRecord]:
    """ToughTables/2T_WD: CSV tables (generic headers, noisy cells) + multi-valid-type
    CTA gold. CTA gt rows = (table_id, col, space-sep acceptable class URLs). We expose
    gold_cta_accept (the SET) for multi-gold cscore; gold_cta = first member as a primary.
    Candidates are NOT seeded here (ToughTables hardness is linking -> use real candgen).
    """
    import csv

    root = dataset_dir("toughtables") / "2T_WD"
    tables_dir = root / "tables"
    cta_gt: dict[str, dict[int, set[str]]] = {}
    with open(root / "gt" / "CTA_2T_WD_gt.csv", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            tid, col, urls = row[0], int(row[1]), row[2]
            cls = {q for q in (_wd_id(u) for u in urls.split()) if q}
            cta_gt.setdefault(tid, {})[col] = cls

    for tid in sorted(cta_gt):
        fp = tables_dir / f"{tid}.csv"
        if not fp.exists():
            continue
        with open(fp, newline="") as f:
            rows_all = list(csv.reader(f))
        if not rows_all:
            continue
        headers = rows_all[0]
        rows = rows_all[1:]                       # first row is the (generic) header
        ncols = len(headers)
        accept = {c: s for c, s in cta_gt[tid].items() if c < ncols}
        gold_cta = {c: next(iter(s)) for c, s in accept.items() if s}
        yield STIRecord(
            table_id=tid,
            headers=[str(h) for h in headers],
            rows=[[str(x) for x in r] + [""] * (ncols - len(r)) for r in rows],
            entity_columns=sorted(accept),
            gold_cta=gold_cta,
            gold_cpa={},                          # ToughTables has no CPA gt
            gold_cea={},                          # use real candgen, not gold CEA
            gold_cta_accept=accept,
        )


def _load_250wt() -> Iterator[STIRecord]:
    """Load 250WT via the canonical `sm` semantic-model ground truth.

    CTA gold: from SemanticModel.get_semantic_types_of_column (class attached to a
    column via rdfs:label). CEA gold: from the table link matrix. CPA gold: ClassNode
    -> ClassNode edges, resolving one level of wikibase:Statement reification; the
    relationship property is normalised to a bare Pxxx id.
    """
    from sm.dataset import Dataset  # local import: heavy dependency, only for this loader

    ds_path = dataset_dir("250wt")
    examples = Dataset(ds_path).load()
    for ex in examples:
        ft = ex.table
        cols = ft.table.columns
        headers = [c.name or "" for c in cols]
        nrows = ft.nrows()
        rows = [[str(cols[c].values[r]) for c in range(len(cols))] for r in range(nrows)]

        # Prefer the first (primary) semantic model; 250WT ships >=1 per table.
        sm0 = ex.sms[0]

        # --- CTA + the set of class nodes that stand for a column ----------------
        gold_cta: dict[int, str] = {}
        for c in cols:
            sts = sm0.get_semantic_types_of_column(c.index)
            for st in sts:
                cid = _wd_id(st.class_abs_uri)
                if cid is not None:
                    gold_cta[c.index] = cid
                    break  # one type per column (SemTab convention); first wins

        # Two ways a graph node maps to a table column:
        #  - a ClassNode is the *type of* a column iff it has an rdfs:label edge to
        #    that column's DataNode (these edges define CTA, not CPA);
        #  - a DataNode *is* a column directly (entity OR literal column).
        col_of_classnode: dict[int, int] = {}
        for e in sm0.iter_edges():
            if e.abs_uri == _RDFS_LABEL:
                dn = sm0.get_node(e.target)
                col_index = getattr(dn, "col_index", None)
                if col_index is not None:
                    col_of_classnode[e.source] = col_index
        col_of_datanode: dict[int, int] = {
            n.id: n.col_index
            for n in sm0.iter_nodes()
            if getattr(n, "col_index", None) is not None
        }

        def _col_of(node_id: int) -> int | None:
            if node_id in col_of_classnode:
                return col_of_classnode[node_id]
            return col_of_datanode.get(node_id)

        # --- CPA: relationships subject-column -> object-column ------------------
        # Covers entity->entity AND entity->literal columns; resolves one level of
        # wikibase:Statement reification. rdfs:label edges are CTA, excluded here.
        gold_cpa: dict[tuple[int, int], str] = {}
        cpa_via_statement: dict[tuple[int, int], dict] = {}
        for e in sm0.iter_edges():
            if e.source not in col_of_classnode or e.abs_uri == _RDFS_LABEL:
                continue
            subj_col = col_of_classnode[e.source]
            prop = _wd_id(e.abs_uri)
            tgt = sm0.get_node(e.target)
            if getattr(tgt, "abs_uri", None) == _STATEMENT:  # reified n-ary
                for e2 in sm0.iter_edges():
                    if e2.source != e.target or e2.abs_uri == _RDFS_LABEL:
                        continue
                    obj_col = _col_of(e2.target)
                    if obj_col is None or obj_col == subj_col:
                        continue
                    qual = _wd_id(e2.abs_uri)
                    rel = qual or prop  # convention flagged for later reconciliation
                    if rel:
                        gold_cpa[(subj_col, obj_col)] = rel
                        cpa_via_statement[(subj_col, obj_col)] = {
                            "statement_property": prop,
                            "qualifier_property": qual,
                        }
            else:  # direct class->class or class->literal-column
                obj_col = _col_of(e.target)
                if prop and obj_col is not None and obj_col != subj_col:
                    gold_cpa[(subj_col, obj_col)] = prop

        # --- CEA gold + entity columns from the link matrix ---------------------
        gold_cea: dict[tuple[int, int], str] = {}
        for r in range(nrows):
            for c in range(len(cols)):
                links = ft.links[r, c]
                if links and getattr(links[0], "entities", None):
                    gold_cea[(r, c)] = str(links[0].entities[0])

        entity_columns = sorted(
            set(gold_cta) | {c for (_, c) in gold_cea}
        )

        yield STIRecord(
            table_id=ft.table.table_id,
            headers=headers,
            rows=rows,
            entity_columns=entity_columns,
            gold_cta=gold_cta,
            gold_cpa=gold_cpa,
            gold_cea=gold_cea,
            cpa_via_statement=cpa_via_statement,
        )


def assert_no_250wt_leakage(train_article_ids: set[str], wt250_article_ids: set[str]) -> None:
    """Fail loudly if any 250WT source article leaked into training (CLAUDE.md s2)."""
    overlap = train_article_ids & wt250_article_ids
    if overlap:
        raise AssertionError(
            f"LEAKAGE: {len(overlap)} 250WT articles in train set, e.g. {list(overlap)[:5]}"
        )


# --- CEA candidate interface (DATA_SPEC.md Part A) -------------------------------
@dataclass
class Candidate:
    """One candidate entity for a cell. Feeds cell_candidate nodes (SCHEMA.md)."""

    entity_id: str
    label: str
    type_ids: list[str]                  # instance_of -> instance_of edges
    surface_feats: dict                  # 4 scalars (Levenshtein/JW/MongeElkan/Jaccard)
    context_feats: dict                  # header-desc dot product, property-match count
    prior: float                         # normalized log pagerank
    statements: list[tuple]              # (property_id, object_entity_id) -> kg_triple edges


def generate_candidates(record: "STIRecord", kg_index, encoder, k: int = 100) -> dict:  # noqa: ANN001
    """DATA_SPEC A.1-A.3: return {(row, col): list[Candidate]} for entity columns.

    Uses the CACHED local Wikidata index (never the live API at train/eval time).
    Computes both header-present and header-absent context features so the
    header-dropout robustness perturbation needs no recomputation (A.3).
    """
    raise NotImplementedError("Gate 2: implement candidate retrieval + featurization (A)")
