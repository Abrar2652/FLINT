"""Warm the candidate cache for all 250WT entity-column cell mentions (online,
polite, rate-limited). Idempotent: only fetches misses. Run on a node with net."""
from flint.data.loaders import load_dataset
from flint.data.candgen import CachedCandidateGenerator

recs = list(load_dataset("250wt"))
mentions = []
for r in recs:
    for col in r.entity_columns:
        for row in range(len(r.rows)):
            if col < len(r.rows[row]):
                m = (r.rows[row][col] or "").strip()
                if m:
                    mentions.append(m)
mentions = list(dict.fromkeys(mentions))
print(f"unique entity-column mentions: {len(mentions)}", flush=True)
cg = CachedCandidateGenerator(offline=False)
n = cg.warm_concurrent(mentions, workers=6)
print(f"warmed {n} new mentions", flush=True)
