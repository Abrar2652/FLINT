"""High-recall CPA candidate generation — measure the recall lift.

Current FLINT-CPA relation candidates: property p is a candidate for ordered pair
(i,j) iff the subject entity has a statement (p, obj) with obj == object-cell GOLD
entity (entity-match) OR a literal statement value matching the object cell
(literal-match). This caps CPA recall (250WT ~0.60-0.73, HardTables 0.40, Wikidata 0.34)
and is the binding bottleneck (HeTa-Graph's L_constraint does NOT touch it).

We measure candidate recall (gold property in the candidate set) under progressively
enriched generation:
  R0 baseline            = QID entity-match + current literal-match
  R1 +improved literal   = relaxed numeric (rel-tol) / date (Y-M-D or year) / string-norm
  R2 +entity label-match = statement obj LABEL matches object-cell text (recovers
                           CEA-vs-statement QID mismatches), labels from cache
  R3 +presence (upper)   = property present in subject statements at all (ignores object)
Run: python scripts/flint_candgen_recall.py --dataset 250wt
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

from flint.data.kg import CachedWikidataKG
from flint.data.loaders import load_dataset

_num = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_str(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def lit_match_improved(cell, val, kind):
    cell = (cell or "").strip()
    if not cell or not val:
        return False
    if kind == "quantity":
        cm = _num.findall(cell.replace(",", ""))
        vm = _num.findall(str(val).replace(",", ""))
        if not cm or not vm:
            return False
        try:
            a, b = float(cm[0]), float(vm[0])
        except ValueError:
            return False
        if a == b:
            return True
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom < 0.02            # 2% relative tolerance
    if kind == "time":
        cy, vy = re.findall(r"\d{4}", cell), re.findall(r"\d{4}", str(val))
        return bool(cy and vy and cy[0] == vy[0])
    c, v = norm_str(cell), norm_str(val)
    return c == v or (len(c) >= 3 and (c in v or v in c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="250wt")
    ap.add_argument("--maxrows", type=int, default=200)
    args = ap.parse_args()
    kg = CachedWikidataKG(offline=True)
    recs = [r for r in load_dataset(args.dataset) if r.gold_cpa]

    def label(q):
        try:
            lb = kg.label(q)
        except Exception:  # noqa: BLE001
            return None
        return lb if lb and lb != q else None  # None => uncached/no label

    tot = 0
    hit = {"R0": 0, "R1": 0, "R2": 0, "R3": 0}
    need_labels = 0
    for r in recs:
        nr = min(len(r.rows), args.maxrows)
        for (i, j), gp in r.gold_cpa.items():
            tot += 1
            c0, c1, c2, c3 = set(), set(), set(), set()
            for row in range(nr):
                e = r.gold_cea.get((row, i))
                if not e:
                    continue
                o = r.gold_cea.get((row, j))
                cell = r.rows[row][j] if j < len(r.rows[row]) else ""
                for p, obj in kg.statements(e):
                    c3.add(p)                                   # presence
                    if o and obj == o:
                        c0.add(p); c1.add(p); c2.add(p)         # QID entity-match
                    elif cell.strip():
                        lb = label(obj)                          # label-match
                        if lb and lit_match_improved(cell, lb, "string"):
                            c2.add(p)
                for p, val, kind in kg.literal_statements(e):
                    c3.add(p)
                    # baseline literal-match (original simple rule, approximated)
                    if _basic_lit(cell, val, kind):
                        c0.add(p); c1.add(p); c2.add(p)
                    elif lit_match_improved(cell, val, kind):
                        c1.add(p); c2.add(p)                     # improved literal
            if gp in c0:
                hit["R0"] += 1
            if gp in c1:
                hit["R1"] += 1
            if gp in c2:
                hit["R2"] += 1
            if gp in c3:
                hit["R3"] += 1

    print("=" * 60)
    print(f"{args.dataset} CPA candidate-recall lift (gold pairs={tot})")
    print(f"  R0 baseline (QID + basic literal)        : {hit['R0']/tot:.3f}")
    print(f"  R1 +improved literal match               : {hit['R1']/tot:.3f}")
    print(f"  R2 +entity label-match (cached labels)   : {hit['R2']/tot:.3f}")
    print(f"  R3 +presence (property-present, UPPER bnd): {hit['R3']/tot:.3f}")
    print("=" * 60)


def _basic_lit(cell, val, kind):
    from flint.data.kg import literal_matches
    return literal_matches(cell, val, kind)


if __name__ == "__main__":
    main()
