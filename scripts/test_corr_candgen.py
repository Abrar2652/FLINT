"""Drift-robust CPA candidate generation via VALUE CORRELATION (test).

Hypothesis: synthetic-CPA recall collapses because exact object-value match fails
under snapshot drift (2026 cache vs 2022 benchmark). But for NUMERIC object columns,
the subject entities' values for the gold property still RANK-CORRELATE with the
column values down the rows (e.g. population: absolute values drift, ranking holds).
So: property p is a candidate for (subj_col i, numeric_col j) if the per-row series
[ KG value of p for subject entity ]  vs  [ object cell value ]  correlate highly,
WITHOUT needing exact match. This is snapshot/unit robust and lightweight.

Measure on HardTables/WikidataTables:
 - of gold CPA pairs with a NUMERIC object column, what fraction does the gold
   property's |Spearman corr| exceed a threshold (recoverable), and is it the TOP
   correlating property (uniquely identifiable -> precision)?
 - resulting candidate recall vs the exact-match baseline.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

from flint.data.kg import CachedWikidataKG
from flint.data.loaders import load_dataset

_NUM = re.compile(r"-?\d[\d.]*")


def num(s):
    m = _NUM.findall((s or "").replace(",", ""))
    try:
        return float(m[0]) if m else None
    except ValueError:
        return None


def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    vy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hardtables")
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--maxrows", type=int, default=200)
    args = ap.parse_args()
    kg = CachedWikidataKG(offline=True)
    recs = [r for r in load_dataset(args.dataset) if r.gold_cpa]

    numeric_pairs = exact_hit = corr_hit = corr_top = 0
    for r in recs:
        nr = min(len(r.rows), args.maxrows)
        for (i, j), gp in r.gold_cpa.items():
            # object column numeric?
            colvals = [num(r.rows[row][j]) for row in range(nr) if j < len(r.rows[row])]
            frac_num = sum(v is not None for v in colvals) / max(len(colvals), 1)
            if frac_num < 0.7:
                continue  # not a numeric column
            numeric_pairs += 1
            # gather per-property series: for each subject entity row, its quantity values
            # build column series of cell numbers paired with each property's KG value
            prop_series = defaultdict(list)   # p -> list of (row_cell_num, kg_val)
            exact_props = set()
            for row in range(nr):
                e = r.gold_cea.get((row, i))
                cell = num(r.rows[row][j]) if j < len(r.rows[row]) else None
                if not e or cell is None:
                    continue
                seen_p = {}
                for p, val, kind in kg.literal_statements(e):
                    if kind == "quantity":
                        kv = num(val)
                        if kv is not None and p not in seen_p:
                            seen_p[p] = kv
                for p, kv in seen_p.items():
                    prop_series[p].append((cell, kv))
                    from flint.data.kg import literal_matches
                    if literal_matches(r.rows[row][j], val if False else str(kv), "quantity"):
                        pass
            # exact-match recall (baseline): gold p present with a matching value
            for row in range(nr):
                e = r.gold_cea.get((row, i))
                cell = r.rows[row][j] if j < len(r.rows[row]) else ""
                if not e:
                    continue
                for p, val, kind in kg.literal_statements(e):
                    if kind == "quantity" and p == gp:
                        a, b = num(cell), num(val)
                        if a is not None and b is not None and abs(a - b) < 1e-6:
                            exact_props.add(p)
            if gp in exact_props:
                exact_hit += 1
            # correlation recall: gold p's series correlates
            corrs = {}
            for p, series in prop_series.items():
                if len(series) >= 3:
                    cs = [s[0] for s in series]
                    ks = [s[1] for s in series]
                    corrs[p] = abs(spearman(cs, ks))
            if corrs.get(gp, 0) >= args.thr:
                corr_hit += 1
                if gp == max(corrs, key=corrs.get):
                    corr_top += 1

    print("=" * 60)
    print(f"{args.dataset} CPA — numeric object columns: {numeric_pairs} gold pairs")
    if numeric_pairs:
        print(f"  exact-value-match recall (baseline)      : {exact_hit/numeric_pairs:.3f}")
        print(f"  correlation recall (|rho|>={args.thr})        : {corr_hit/numeric_pairs:.3f}")
        print(f"  correlation AND gold is TOP-correlated    : {corr_top/numeric_pairs:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
