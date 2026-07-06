"""Distant-supervision labeling pipeline (DATA_SPEC.md Part B).

Turns raw Wikipedia tables into training examples via in-table hyperlinks.
STATUS: STUB keyed to DATA_SPEC.md section numbers. Filled across Gate 2/3.

Kept strictly separate from model code: the greedy CTA labeler here is a
LABEL-GENERATION heuristic on high-confidence linked entities, NOT the model's
learned CTA. Do not import model code into this module.
"""
from __future__ import annotations

from .loaders import STIRecord, assert_no_250wt_leakage  # noqa: F401  (re-export for pipeline)


def is_easy_to_label(wiki_table) -> bool:  # noqa: ANN001
    """DATA_SPEC B.1: >=10 rows, <=1 link/cell, a column >=70% linked, >80% links resolve.

    Each of the four conditions is asserted individually and logged so the filter
    is auditable (CLAUDE.md s2).
    """
    raise NotImplementedError("Gate 2: implement the four easy-to-label conditions (B.1)")


def clean_inconsistent_links(table, blocklist):  # noqa: ANN001
    """DATA_SPEC B.2: drop links whose column header is incompatible with the
    column's provisional (most-common) entity type. `blocklist` is loaded from the
    versioned data file data/processed/blocklist.tsv, not hard-coded."""
    raise NotImplementedError("Gate 2: implement blocklist-based link cleaning (B.2)")


def label_cta(table, kg_index) -> dict[int, str]:  # noqa: ANN001
    """DATA_SPEC B.3: CTA gold per column from hyperlink-resolved entities via the
    greedy ancestor heuristic (label-generation only)."""
    raise NotImplementedError("Gate 3: implement CTA label generation (B.3)")


def label_cpa(table, kg_index):  # noqa: ANN001
    """DATA_SPEC B.4: returns (cpa_gold, negatives).

    Match statement values within rows -> candidate relationships (incl. n-ary);
    drop those in <50% rows or with >10% contradicting rows; keep highest-frequency
    property per column-pair as gold; unselected generated relationships -> negatives.
    """
    raise NotImplementedError("Gate 3: implement CPA label generation (B.4)")


def build_training_examples(wiki_dump, kg_index, n: int = 5000, *, wt250_ids: set[str]):  # noqa: ANN001
    """DATA_SPEC B.1->B.6: full pipeline.

    MUST call assert_no_250wt_leakage(...) before returning (B.5), draw the n-table
    sample with a fixed seed, write chosen ids to data/processed/train_table_ids.txt,
    and attach per-label confidence proxies (B.6) for the calibration study.
    """
    raise NotImplementedError("Gate 3: assemble B.1-B.6, enforce leakage guard")
