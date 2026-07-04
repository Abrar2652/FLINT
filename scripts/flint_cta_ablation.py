"""CTA feature ablation for FLINT: leave-one-feature-out, measure cscore drop.

Replicates flint_cta_ranker's feature construction EXACTLY (same 11 features, same
order, same blended cscore+exact target, 5-fold CV) but, for each feature, removes
that column from the design matrix and re-runs, reporting the micro-cscore delta.
Seed 0 only (the ranker std is +/-0.001, so a single fold-seed is representative).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor

from flint.data.kg import CachedWikidataKG, ancestors_with_hops, hierarchy_distance, is_ancestor
from flint.data.loaders import load_dataset
from flint.data.splits import frozen_kfold
from flint.eval.evaluate import evaluate_cta
from flint.eval.metrics import cscore
from flint.graph.features import SentenceTransformerEncoder

MAX_HOPS = 4
COV_FLOOR = 0.2
TYPE_PROPS = {"P106", "P641", "P39", "P101", "P452", "P136", "P413"}
FEATURE_NAMES = ["anc_coverage", "direct_coverage", "cov_rank", "is_mode",
                 "is_ancestor_of_mode", "hops_above_mode", "n_ancestors",
                 "header_cos", "context_cos", "entity_cos", "freq_prior"]


def main() -> None:
    kg = CachedWikidataKG(offline=True)
    recs = list(load_dataset("250wt"))
    gold = {r.table_id: r.gold_cta for r in recs}
    enc = SentenceTransformerEncoder()
    lab = lambda q: (kg.label(q) or q)  # noqa: E731

    cols, labset, entset = [], set(), set()
    for r in recs:
        for col in r.entity_columns:
            ents = [e for row in range(len(r.rows)) if (e := r.gold_cea.get((row, col)))]
            if not ents:
                continue
            clos = [{t for d in kg.instance_of(e) for t in ancestors_with_hops(kg, d, max_hops=MAX_HOPS)}
                    for e in ents]
            clos = [c for c in clos if c]
            n = len(clos) or 1
            cnt = Counter(t for c in clos for t in c)
            cov = {t: cnt[t] / n for t in cnt}
            dcnt = Counter()
            for e in ents:
                for d in set(kg.instance_of(e)):
                    dcnt[d] += 1
            dcov = {t: dcnt[t] / n for t in dcnt}
            direct = Counter(t for e in ents for t in kg.instance_of(e))
            if not direct:
                continue
            mode = direct.most_common(1)[0][0]
            cands = set(t for t, c in cov.items() if c >= COV_FLOOR)
            for e in ents:
                for p, o in kg.statements(e):
                    if p in TYPE_PROPS and o.startswith("Q"):
                        cands.update(ancestors_with_hops(kg, o, max_hops=2))
            cands = list(cands) or [mode]
            header = r.headers[col] if col < len(r.headers) else ""
            sample = [r.rows[row][col] for row in range(min(8, len(r.rows))) if col < len(r.rows[row])]
            ctx = (header + " : " + ", ".join(sample))[:256]
            es = list(dict.fromkeys(ents))[:10]
            cols.append(dict(tid=r.table_id, col=col, cov=cov, dcov=dcov, mode=mode,
                             cands=cands, header=header, ctx=ctx, ents=es))
            labset.update(cands)
            entset.update(es)

    labs = sorted(labset); L = {q: i for i, q in enumerate(labs)}
    lab_emb = F.normalize(enc.encode([lab(q) for q in labs]), dim=1)
    hdrs = sorted({c["header"] for c in cols}); H = {h: i for i, h in enumerate(hdrs)}
    hdr_emb = F.normalize(enc.encode(hdrs), dim=1)
    ctxs = sorted({c["ctx"] for c in cols}); C = {c: i for i, c in enumerate(ctxs)}
    ctx_emb = F.normalize(enc.encode(ctxs), dim=1)
    el = sorted(entset); E = {e: i for i, e in enumerate(el)}
    ent_emb = F.normalize(enc.encode([lab(e) for e in el]), dim=1)
    for ci in cols:
        idx = [E[e] for e in ci["ents"] if e in E]
        ci["em"] = F.normalize(ent_emb[idx].mean(0), dim=0) if idx else None

    nac: dict[str, int] = {}
    def n_anc(t):
        if t not in nac:
            nac[t] = len(ancestors_with_hops(kg, t, max_hops=6))
        return nac[t]

    def feats(ci, t, prior, N):
        cov, dcov, mode, cands = ci["cov"], ci["dcov"], ci["mode"], ci["cands"]
        c = cov.get(t, 0.0)
        rank = sum(1 for u in cands if cov.get(u, 0) > c) / max(len(cands), 1)
        anc = float(is_ancestor(kg, t, mode))
        d = hierarchy_distance(kg, t, mode) if anc else 0
        hco = float(hdr_emb[H[ci["header"]]] @ lab_emb[L[t]])
        xco = float(ctx_emb[C[ci["ctx"]]] @ lab_emb[L[t]])
        eco = float(ci["em"] @ lab_emb[L[t]]) if ci["em"] is not None else 0.0
        pr = math.log(prior.get(t, 0) / N + 1e-9)
        return [c, dcov.get(t, 0.0), rank, float(t == mode), anc, float(d or 0),
                float(n_anc(t)), hco, xco, eco, pr]

    isa = lambda a, b: is_ancestor(kg, a, b)  # noqa: E731
    hd = lambda a, b: hierarchy_distance(kg, a, b)  # noqa: E731
    folds = frozen_kfold(sorted({r.table_id for r in recs}), k=5, seed=0)

    def run(drop):
        pred = {r.table_id: {} for r in recs}
        for test in folds:
            ts = set(test)
            prior = Counter()
            for r in recs:
                if r.table_id not in ts:
                    for g in r.gold_cta.values():
                        if g:
                            prior[g] += 1
            N = sum(prior.values()) or 1
            X, yc, ye = [], [], []
            for ci in cols:
                if ci["tid"] in ts:
                    continue
                g = gold[ci["tid"]].get(ci["col"])
                if not g:
                    continue
                for t in ci["cands"]:
                    X.append(feats(ci, t, prior, N))
                    yc.append(cscore(t, g, hierarchy_distance=hd, is_ancestor=isa))
                    ye.append(1.0 if t == g else 0.0)
            Xa = np.array(X)
            if drop is not None:
                Xa = np.delete(Xa, drop, axis=1)
            rc = HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.06,
                                               min_samples_leaf=30, l2_regularization=1.0).fit(Xa, np.array(yc))
            re = HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.06,
                                               min_samples_leaf=30, l2_regularization=1.0).fit(Xa, np.array(ye))
            for ci in cols:
                if ci["tid"] not in ts:
                    continue
                xs = np.array([feats(ci, t, prior, N) for t in ci["cands"]])
                if drop is not None:
                    xs = np.delete(xs, drop, axis=1)
                s = rc.predict(xs) + re.predict(xs)
                pred[ci["tid"]][ci["col"]] = ci["cands"][int(s.argmax())]
        return evaluate_cta(pred, gold, kg)["cta_micro_f1"]

    full = run(None)
    print("=" * 56)
    print(f"CTA feature ablation (250WT, gold, seed0, micro cscore)")
    print(f"  FULL model: {full:.3f}")
    print("-" * 56)
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        f = run(i)
        rows.append((name, f, full - f))
        print(f"  -{name:20s}: {f:.3f}  (drop {full - f:+.3f})")
    print("-" * 56)
    rows.sort(key=lambda x: -x[2])
    print("  most important (largest drop):", ", ".join(f"{n}" for n, _, _ in rows[:3]))
    print("=" * 56)


if __name__ == "__main__":
    main()
