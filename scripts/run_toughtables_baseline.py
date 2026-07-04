"""ToughTables baseline (Stage-3 diagnostic): does a counting baseline with REAL
candidates reproduce SOTA-like difficulty (~0.41-0.54)? If so, the frontier is real
in our setup and there is headroom for the Graph Transformer to win.

Pipeline: sample rows -> candgen (mention->entities) -> KG warm (entity->P31) ->
counting baseline (majority P31 of each cell's top-1 candidate) -> multi-gold cscore.

Usage: python scripts/run_toughtables_baseline.py --rows 20 --k 20 --workers 12
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

from flint.data.candgen import CachedCandidateGenerator
from flint.data.kg import CachedWikidataKG
from flint.data.loaders import load_dataset
from flint.eval.evaluate import evaluate_cta_multigold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20, help="rows sampled per table (CTA needs few)")
    ap.add_argument("--k", type=int, default=20, help="candidates per cell")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    recs = list(load_dataset("toughtables"))
    if args.limit:
        recs = recs[: args.limit]
    cg = CachedCandidateGenerator(k=args.k)
    kg = CachedWikidataKG()

    # sampled (table,col,row) entity cells
    cells = []  # (tid, col, mention)
    for r in recs:
        nr = min(len(r.rows), args.rows)
        for col in r.entity_columns:
            for row in range(nr):
                if col < len(r.rows[row]):
                    cells.append((r.table_id, col, r.rows[row][col]))
    mentions = sorted({m for _, _, m in cells if m.strip()})
    print(f"{len(recs)} tables, {len(cells)} sampled entity cells, {len(mentions)} unique mentions",
          flush=True)

    # Phase A: candgen warm
    t0 = time.time()
    n = cg.warm_concurrent(mentions, workers=args.workers)
    print(f"[A] candgen: {n} mentions fetched, {time.time()-t0:.0f}s", flush=True)

    # Phase B: top-1 candidate per cell -> KG warm those entities
    top1 = {}
    ents = set()
    for m in mentions:
        c = cg.candidates(m, k=args.k)
        if c:
            top1[m] = c[0]
            ents.add(c[0])
    t0 = time.time()
    n = kg.warm_concurrent(sorted(ents), workers=args.workers)
    print(f"[B] KG: {len(ents)} entities, {n} fetched, {time.time()-t0:.0f}s", flush=True)

    # counting baseline: majority P31 of top-1 candidates per (table,col)
    kg_off = CachedWikidataKG(offline=True)
    pred = {}
    for r in recs:
        nr = min(len(r.rows), args.rows)
        cp = {}
        for col in r.entity_columns:
            cnt = Counter()
            for row in range(nr):
                if col >= len(r.rows[row]):
                    continue
                m = r.rows[row][col]
                e = top1.get(m)
                if e:
                    for t in kg_off.instance_of(e):
                        cnt[t] += 1
            if cnt:
                cp[col] = cnt.most_common(1)[0][0]
        pred[r.table_id] = cp
    accept = {r.table_id: r.gold_cta_accept for r in recs}
    m = evaluate_cta_multigold(pred, accept, kg_off)
    print("\n=== ToughTables COUNTING baseline (real candidates, multi-gold cscore) ===")
    print(f"  CTA micro-F1 = {m['cta_micro_f1']:.3f}  macro-F1 = {m['cta_macro_f1']:.3f}")
    print(f"  (SOTA: DAGOBAH 0.409, KGCODE-Tab 0.543) -> headroom check)")


if __name__ == "__main__":
    main()
