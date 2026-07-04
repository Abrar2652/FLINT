"""Pre-warm the Wikidata cache for 250WT so per-table graph builds are instant.

Three concurrent phases: (1) gold entities, (2) their P31 types, (3) the P279
ancestor closure (BFS, `--max-hops`). Idempotent: only missing entities fetched.

Usage:
    python scripts/prewarm_cache.py --limit 80        # first subset
    python scripts/prewarm_cache.py                    # all 250WT
"""
from __future__ import annotations

import argparse
import time

from flint.data.kg import CachedWikidataKG
from flint.data.loaders import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="250wt", help="250wt|hardtables|wikidatatables|...")
    ap.add_argument("--limit", type=int, default=0, help="first N tables (0 = all)")
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    recs = list(load_dataset(args.dataset))
    if args.limit:
        recs = recs[: args.limit]
    kg = CachedWikidataKG()

    # phase 1: entities (+ gold CTA types so eval has their ancestor chains)
    ents = sorted({e for r in recs for e in r.gold_cea.values()}
                  | {t for r in recs for t in r.gold_cta.values()})
    t0 = time.time()
    n1 = kg.warm_concurrent(ents, workers=args.workers)
    print(f"[1/3] entities: {len(ents)} total, {n1} fetched, {time.time()-t0:.0f}s", flush=True)

    # phase 2: their P31 types
    types = sorted({t for e in ents for t in kg.instance_of(e)})
    t0 = time.time()
    n2 = kg.warm_concurrent(types, workers=args.workers)
    print(f"[2/3] types: {len(types)} total, {n2} fetched, {time.time()-t0:.0f}s", flush=True)

    # phase 3: P279 ancestor closure (BFS by level)
    frontier = set(types)
    seen = set(types)
    total3 = 0
    for hop in range(args.max_hops):
        parents = sorted({p for c in frontier for p in kg.subclass_of(c)} - seen)
        if not parents:
            break
        t0 = time.time()
        n = kg.warm_concurrent(parents, workers=args.workers)
        total3 += n
        print(f"[3/3] hop {hop+1}: {len(parents)} classes, {n} fetched, {time.time()-t0:.0f}s",
              flush=True)
        seen.update(parents)
        frontier = set(parents)
    print(f"DONE. cache now has entities+types+ancestors for {len(recs)} tables.", flush=True)


if __name__ == "__main__":
    main()
