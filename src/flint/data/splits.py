"""Frozen, hashed dataset splits (CLAUDE.md s2: write once, re-load, never re-randomize).

For the Gate-3 supervised-learnability milestone we k-fold 250WT itself (train on
folds, eval held-out). NOTE: this is NOT GRAMS+'s protocol (they train on distant
supervision and evaluate all of 250WT) — so a CV number here is a learnability/
ceiling diagnostic, not a head-to-head GRAMS+ comparison. The distant-supervision
protocol is the true Gate-3 pass condition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flint.utils.paths import processed_dir


def _seeded_permutation(items: list[str], seed: int) -> list[str]:
    """Deterministic permutation by hashing (item, seed) — no global RNG state."""
    return sorted(items, key=lambda x: hashlib.blake2b(f"{seed}:{x}".encode()).hexdigest())


def make_kfold(table_ids: list[str], *, k: int = 5, seed: int = 0) -> list[list[str]]:
    """Return k disjoint folds (lists of table ids), deterministic in (ids, k, seed)."""
    perm = _seeded_permutation(sorted(table_ids), seed)
    return [perm[i::k] for i in range(k)]


def frozen_kfold(
    table_ids: list[str], *, k: int = 5, seed: int = 0, name: str = "250wt"
) -> list[list[str]]:
    """k-fold split persisted to data/processed/splits/<name>_k{k}_s{seed}.json.

    Written once with a content hash; re-loaded thereafter (asserts the id set
    matches, so a changed corpus fails loudly instead of silently re-splitting).
    """
    splits_dir = processed_dir() / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    fp = splits_dir / f"{name}_k{k}_s{seed}.json"
    id_hash = hashlib.blake2b(",".join(sorted(table_ids)).encode()).hexdigest()[:16]

    if fp.exists():
        blob = json.loads(fp.read_text())
        if blob["id_hash"] != id_hash:
            raise AssertionError(
                f"corpus changed since split was frozen ({fp}); delete it to re-create"
            )
        return blob["folds"]

    folds = make_kfold(table_ids, k=k, seed=seed)
    fp.write_text(json.dumps({"id_hash": id_hash, "k": k, "seed": seed, "folds": folds}))
    return folds
