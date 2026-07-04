"""ToughTables counting baseline from the WARM cache only (offline). Fast diagnostic."""
from __future__ import annotations
import time
from collections import Counter
from flint.data.candgen import CachedCandidateGenerator
from flint.data.kg import CachedWikidataKG
from flint.data.loaders import load_dataset
from flint.eval.evaluate import evaluate_cta_multigold

t0 = time.time()
recs = list(load_dataset("toughtables"))
print(f"loaded {len(recs)} tables {time.time()-t0:.0f}s", flush=True)
cg = CachedCandidateGenerator(k=20, offline=True)
kg = CachedWikidataKG(offline=True)
pred = {}
covered_cells = total_cells = 0
for r in recs:
    nr = min(len(r.rows), 20)
    cp = {}
    for col in r.entity_columns:
        cnt = Counter()
        for row in range(nr):
            if col < len(r.rows[row]):
                total_cells += 1
                c = cg.candidates(r.rows[row][col], k=20)
                if c:
                    covered_cells += 1
                    for tp in kg.instance_of(c[0]):
                        cnt[tp] += 1
        if cnt:
            cp[col] = cnt.most_common(1)[0][0]
    pred[r.table_id] = cp
accept = {r.table_id: r.gold_cta_accept for r in recs}
m = evaluate_cta_multigold(pred, accept, kg)
print(f"candidate coverage: {covered_cells}/{total_cells} cells have >=1 cached candidate", flush=True)
print(f"RESULT ToughTables counting (real candidates): CTA micro-F1={m['cta_micro_f1']:.3f} "
      f"macro={m['cta_macro_f1']:.3f}  (SOTA DAGOBAH 0.409 / KGCODE 0.543)", flush=True)
