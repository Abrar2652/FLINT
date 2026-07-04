"""Warm a 2023-06-19 Wikidata snapshot for the entities of a dataset, via the
MediaWiki revision API (revision active as of the date). Writes trimmed entity
JSON to data/cache/wikidata_2023/ in the SAME format CachedWikidataKG reads, so
FLINT can run offline against the snapshot GRAMS+/MTab/DAGOBAH actually used.

This is the data fix for synthetic-CPA recall (gold relation values exist in 2023,
not in our 2026 cache). Targeted (only dataset entities), concurrent, rate-limited.

Usage: python scripts/warm_2023_snapshot.py --dataset hardtables --limit 150 --workers 8
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from flint.data.kg import CachedWikidataKG
from flint.data.loaders import load_dataset
from flint.utils.paths import cache_dir

DATE = "2023-06-19T00:00:00Z"
API = "https://www.wikidata.org/w/api.php"
UA = {"User-Agent": "flint-research/0.1 (semantic table interpretation research)"}
OUT = cache_dir() / "wikidata_2023"
OUT.mkdir(parents=True, exist_ok=True)
_kg = CachedWikidataKG(offline=True)  # for _trim only


def fetch_2023(qid: str) -> bool:
    fp = OUT / f"{qid}.json"
    if fp.exists():
        return False
    for attempt in range(4):
        try:
            q = urllib.parse.urlencode({
                "action": "query", "prop": "revisions", "titles": qid,
                "rvstart": DATE, "rvlimit": 1, "rvdir": "older",
                "rvprop": "content|timestamp", "rvslots": "main", "format": "json"})
            req = urllib.request.Request(f"{API}?{q}", headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                d = json.loads(r.read())
            pages = d.get("query", {}).get("pages", {})
            pg = next(iter(pages.values()))
            revs = pg.get("revisions")
            if not revs:                       # entity didn't exist in 2023
                fp.write_text(json.dumps({"v": _kg.TRIM_VERSION, "P31": [], "P279": [],
                                          "statements": [], "literals": [], "label": qid}))
                return True
            ent = json.loads(revs[0]["slots"]["main"]["*"])
            fp.write_text(json.dumps(_kg._trim(ent)))
            return True
        except Exception:  # noqa: BLE001
            time.sleep(min(2 ** attempt, 12))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hardtables")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    recs = list(load_dataset(args.dataset))
    if args.limit:
        recs = recs[: args.limit]
    ents = sorted({e for r in recs for e in r.gold_cea.values()})
    todo = [e for e in ents if not (OUT / f"{e}.json").exists()]
    print(f"{args.dataset}: {len(ents)} entities, {len(todo)} to fetch @2023", flush=True)
    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, ok in enumerate(ex.map(fetch_2023, todo)):
            n += ok
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: fetched {n}, {time.time()-t0:.0f}s; cache={OUT}", flush=True)


if __name__ == "__main__":
    main()
