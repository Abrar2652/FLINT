"""Official SemTab ToughTables CTA scoring (AH / AP) using the provided
gt / gt_ancestor / gt_descendent files — so our numbers are COMPARABLE to published
SOTA (DAGOBAH 0.409, KGCODE 0.543). Do not substitute our KG hierarchy here; the
official depths are authoritative and avoid snapshot drift.

Scoring (SemTab CTA cscore):
  perfect (pred in the gt set)      -> 1.0
  ancestor at depth d (d<=5)        -> 0.8**d   (predicted too general)
  descendant at depth d (d<=3)      -> 0.7**d   (predicted too specific)
  otherwise                         -> 0
  best credit is taken over the column's perfect-set members.
AH (primary, recall-like) = sum(scores) / #target cols   (missing pred scores 0)
AP (precision-like)       = sum(scores) / #predicted cols
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_QID = re.compile(r"[QP]\d+")


def _qid(s: str) -> str | None:
    m = _QID.search(s)
    return m.group(0) if m else None


def load_official_gt(gt_dir: Path):
    """Return (perfect, ancestor, descendant):
    perfect[(tid,col)] = set(qid); ancestor[gold_qid] = {qid: depth}; descendant same."""
    import csv

    perfect: dict[tuple[str, int], set[str]] = {}
    with open(gt_dir / "CTA_2T_WD_gt.csv", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            tid, col = row[0], int(row[1])
            perfect[(tid, col)] = {q for q in (_qid(u) for u in row[2].split()) if q}

    def _load(name):
        raw = json.load(open(gt_dir / name))
        out = {}
        for k, v in raw.items():
            gk = _qid(k)
            if gk is None:
                continue
            out[gk] = {_qid(ik): int(dv) for ik, dv in v.items() if _qid(ik)}
        return out

    return perfect, _load("CTA_2T_WD_gt_ancestor.json"), _load("CTA_2T_WD_gt_descendent.json")


def ah_credit(pred: str, perfect_set: set[str], ancestor: dict, descendant: dict) -> float:
    """Official partial-credit score for one CTA prediction."""
    if pred in perfect_set:
        return 1.0
    best = 0.0
    for g in perfect_set:
        d = ancestor.get(g, {}).get(pred)
        if d is not None and d <= 5:
            best = max(best, 0.8 ** d)
        d = descendant.get(g, {}).get(pred)
        if d is not None and d <= 3:
            best = max(best, 0.7 ** d)
    return best


def score_official(pred_per_table: dict[str, dict[int, str]], gt_dir: Path) -> dict:
    """AH (primary), AP, and their F1 over all CTA target columns."""
    perfect, anc, desc = load_official_gt(gt_dir)
    total = sum_score = predicted = 0
    for (tid, col), pset in perfect.items():
        if not pset:
            continue
        total += 1
        pred = pred_per_table.get(tid, {}).get(col)
        if pred is None:
            continue
        predicted += 1
        sum_score += ah_credit(pred, pset, anc, desc)
    ah = sum_score / total if total else 0.0          # recall-like (primary)
    ap = sum_score / predicted if predicted else 0.0  # precision-like
    f1 = 0.0 if (ah + ap) == 0 else 2 * ah * ap / (ah + ap)
    return {"AH": ah, "AP": ap, "F1": f1, "n_targets": total, "n_predicted": predicted}
