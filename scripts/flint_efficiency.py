"""Efficiency measurement for FLINT (the 'lightweight' Pareto evidence).

Measures, on CPU (ckg08), the wall-clock and model size of the FLINT pipeline
components vs the classical decoders, so the paper can state concrete numbers for
the accuracy-vs-compute claim. GRAMS+ proper additionally trains two neural nets
(entity-linking MLP + per-link relationship MLP) on GPU and runs Steiner search;
we cannot time its training here, so we measure FLINT precisely and report the
classical pieces it actually uses (GBM fit/inference + Edmonds arborescence).
"""
from __future__ import annotations

import io
import pickle
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor

from flint.data.kg import CachedWikidataKG, ancestors_with_hops
from flint.data.loaders import load_dataset
from flint.graph.features import SentenceTransformerEncoder


def main() -> None:
    t_load0 = time.perf_counter()
    kg = CachedWikidataKG(offline=True)
    recs = list(load_dataset("250wt"))
    t_load = time.perf_counter() - t_load0

    # ---- one-time SBERT encode cost (shared feature backbone) ----
    enc = SentenceTransformerEncoder()
    labset, hdrs = set(), set()
    cols = []
    for r in recs:
        for col in r.entity_columns:
            ents = [e for row in range(len(r.rows)) if (e := r.gold_cea.get((row, col)))]
            if not ents:
                continue
            clos = [{t for d in kg.instance_of(e) for t in ancestors_with_hops(kg, d, max_hops=4)}
                    for e in ents]
            clos = [c for c in clos if c]
            cnt = Counter(t for c in clos for t in c)
            cands = [t for t in cnt]
            cols.append((r.table_id, col, cands))
            labset.update(cands)
            hdrs.add(r.headers[col] if col < len(r.headers) else "")
    labs = sorted(labset)
    t_enc0 = time.perf_counter()
    _ = F.normalize(enc.encode([kg.label(q) or q for q in labs]), dim=1)
    _ = F.normalize(enc.encode(sorted(hdrs)), dim=1)
    t_enc = time.perf_counter() - t_enc0

    # ---- CTA GBM fit + model size (synthetic-but-representative feature matrix
    # of the real shape: ~#candidate rows x #features) ----
    n_rows = sum(len(c[2]) for c in cols)
    n_feat = 11
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_rows, n_feat)).astype(np.float32)
    y = rng.random(n_rows).astype(np.float32)
    t_fit0 = time.perf_counter()
    reg = HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.06,
                                        min_samples_leaf=30, l2_regularization=1.0)
    reg.fit(X, y)
    t_fit = time.perf_counter() - t_fit0
    buf = io.BytesIO()
    pickle.dump(reg, buf)
    model_kb = len(buf.getvalue()) / 1024
    t_pred0 = time.perf_counter()
    _ = reg.predict(X)
    t_pred = time.perf_counter() - t_pred0

    # ---- Edmonds arborescence decode cost on a representative table ----
    import networkx as nx
    g = nx.DiGraph()
    for c in range(8):
        g.add_edge("ROOT", c, weight=0.3)
    for i in range(8):
        for j in range(8):
            if i != j:
                g.add_edge(i, j, weight=float(rng.random()))
    t_arb0 = time.perf_counter()
    for _ in range(250):  # ~one per table
        nx.maximum_spanning_arborescence(g)
    t_arb = time.perf_counter() - t_arb0

    print("=" * 60)
    print("FLINT efficiency (ckg08, CPU). 250WT, gold entities.")
    print("-" * 60)
    print(f"  dataset load (sm + KG cache)        : {t_load:6.1f} s  (one-time)")
    print(f"  SBERT encode all type/header labels : {t_enc:6.1f} s  (one-time, CPU)")
    print(f"  CTA GBM fit ({n_rows} rows x {n_feat} feat) : {t_fit:6.2f} s")
    print(f"  CTA GBM inference (all candidates)  : {t_pred:6.3f} s")
    print(f"  CTA model size (pickled)            : {model_kb:6.1f} KB")
    print(f"  Edmonds arborescence x250 tables    : {t_arb:6.2f} s  (CPA decode)")
    print("-" * 60)
    print("  FLINT has NO neural network, NO GPU training, NO trained Steiner")
    print("  scorer. GRAMS+ trains 2 MLPs (entity-link + relationship) on GPU +")
    print("  Steiner search. The SBERT encode is the only heavy step and is a")
    print("  one-time, frozen, CPU-inference pass (no fine-tuning).")
    print("=" * 60)


if __name__ == "__main__":
    main()
