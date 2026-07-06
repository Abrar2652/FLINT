"""Warm the pending candidate caches (250WT gold mentions + ToughTables entity cells).

Resumable: warm_concurrent skips mentions whose cache file already exists, so re-running
after an interruption continues where it left off. Polite: modest worker count; candgen
backs off on 429. Progress is logged per chunk. Reads live Wikidata (proper User-Agent).

Usage: PYTHONPATH=src python3 scripts/warm_pending_caches.py [250wt|toughtables|both]
"""
from __future__ import annotations

import sys
import time

from flint.data.candgen import CachedCandidateGenerator
from flint.data.loaders import load_dataset


def collect_250wt() -> list[str]:
    ms = []
    for r in load_dataset("250wt"):
        for (row, col), _ in r.gold_cea.items():
            if row < len(r.rows) and col < len(r.rows[row]):
                c = str(r.rows[row][col]).strip()
                if c:
                    ms.append(c)
    return ms


def collect_toughtables() -> list[str]:
    ms = []
    for r in load_dataset("toughtables"):
        for row in range(len(r.rows)):
            for col in r.entity_columns:
                if col < len(r.rows[row]):
                    c = str(r.rows[row][col]).strip()
                    if c:
                        ms.append(c)
    return ms


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    mentions: list[str] = []
    if which in ("250wt", "both"):
        mentions += collect_250wt()
    if which in ("toughtables", "both"):
        mentions += collect_toughtables()
    seen: set[str] = set()
    uniq = [m for m in mentions if not (m in seen or seen.add(m))]
    print(f"[warm] target='{which}'  unique mentions={len(uniq)}", flush=True)

    cg = CachedCandidateGenerator(offline=False)  # live fetch + cache write
    CHUNK, WORKERS = 500, 6
    done = warmed = 0
    t0 = time.time()
    for i in range(0, len(uniq), CHUNK):
        batch = uniq[i:i + CHUNK]
        warmed += cg.warm_concurrent(batch, workers=WORKERS)
        done += len(batch)
        el = time.time() - t0
        rate = done / el if el else 0.0
        eta = (len(uniq) - done) / rate / 60 if rate else 0.0
        print(f"[warm] {done}/{len(uniq)}  new_fetched={warmed}  "
              f"{rate:.1f}/s  elapsed={el/60:.1f}min  eta={eta:.0f}min", flush=True)
    print(f"[warm] DONE target='{which}' new_fetched={warmed} in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
