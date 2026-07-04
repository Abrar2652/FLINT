"""CPA feature ablation for FLINT: leave-one-feature-out feature importance.

Metric = argmax-recall: among reachable gold pairs (gold property in the candidate
set), the fraction where the gold property is the top-scored candidate. This is
decoder/threshold-independent (it isolates the ranker), so it cleanly measures each
feature's contribution. 5-fold CV, seed 0. Feature order matches flint_cpa_ranker.
"""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier

from flint.data.kg import CachedWikidataKG, ancestors_with_hops, literal_matches
from flint.data.loaders import load_dataset
from flint.data.splits import frozen_kfold
from flint.graph.features import SentenceTransformerEncoder
from flint.utils.paths import cache_dir

FEATURE_NAMES = ["entity_support", "literal_support", "total_support", "support_rank",
                 "n_candidates", "subject_coverage", "obj_is_entity", "header_obj_cos",
                 "header_subj_cos", "freq_prior", "is_subject_col"]


def maj_type(kg, r, col):
    c = Counter()
    for row in range(len(r.rows)):
        e = r.gold_cea.get((row, col))
        if e:
            for t in kg.instance_of(e):
                c[t] += 1
    return c.most_common(1)[0][0] if c else None


def main() -> None:
    kg = CachedWikidataKG(offline=True)
    recs = [r for r in load_dataset("250wt") if r.gold_cpa]
    enc = SentenceTransformerEncoder()
    plabels = json.loads((cache_dir() / "property_labels.json").read_text())

    pairs = []
    for r in recs:
        ncol = len(r.headers)
        for i in r.entity_columns:
            n_subj = sum(1 for row in range(len(r.rows)) if r.gold_cea.get((row, i)))
            if not n_subj:
                continue
            for j in range(ncol):
                if j == i:
                    continue
                ent_c, lit_c = Counter(), Counter()
                obj_ent = False
                for row in range(len(r.rows)):
                    e = r.gold_cea.get((row, i))
                    if not e:
                        continue
                    o = r.gold_cea.get((row, j))
                    if o:
                        obj_ent = True
                        for p, obj in kg.statements(e):
                            if obj == o:
                                ent_c[p] += 1
                    cell = r.rows[row][j] if j < len(r.rows[row]) else ""
                    for p, val, kind in kg.literal_statements(e):
                        if literal_matches(cell, val, kind):
                            lit_c[p] += 1
                cands = sorted(set(ent_c) | set(lit_c))
                if not cands:
                    continue
                pairs.append(dict(tid=r.table_id, i=i, j=j, n_subj=n_subj, nrows=len(r.rows),
                                  ent_c=ent_c, lit_c=lit_c, cands=cands, obj_ent=obj_ent,
                                  hobj=r.headers[j] if j < len(r.headers) else "",
                                  hsubj=r.headers[i] if i < len(r.headers) else "",
                                  gold=r.gold_cpa.get((i, j))))
    strength: dict = {}
    for pr in pairs:
        strength[(pr["tid"], pr["i"])] = strength.get((pr["tid"], pr["i"]), 0) + \
            sum(pr["ent_c"].values()) + sum(pr["lit_c"].values())
    best_subj = {}
    for (tid, i), s in strength.items():
        if tid not in best_subj or s > strength[(tid, best_subj[tid])]:
            best_subj[tid] = i

    props = sorted({p for pr in pairs for p in pr["cands"]}); P = {p: k for k, p in enumerate(props)}
    prop_emb = F.normalize(enc.encode([plabels.get(p, p) for p in props]), dim=1)
    hdrs = sorted({pr["hobj"] for pr in pairs} | {pr["hsubj"] for pr in pairs}); H = {h: k for k, h in enumerate(hdrs)}
    hdr_emb = F.normalize(enc.encode(hdrs), dim=1)

    def feats(pr, p, prior, N):
        es = pr["ent_c"].get(p, 0) / pr["n_subj"]
        ls = pr["lit_c"].get(p, 0) / pr["n_subj"]
        supp = min(1.0, (pr["ent_c"].get(p, 0) + pr["lit_c"].get(p, 0)) / pr["n_subj"])
        rank = sum(1 for q in pr["cands"]
                   if (pr["ent_c"].get(q, 0) + pr["lit_c"].get(q, 0)) >
                   (pr["ent_c"].get(p, 0) + pr["lit_c"].get(p, 0))) / max(len(pr["cands"]), 1)
        hco = float(hdr_emb[H[pr["hobj"]]] @ prop_emb[P[p]])
        hcs = float(hdr_emb[H[pr["hsubj"]]] @ prop_emb[P[p]])
        pr_log = math.log(prior.get(p, 0) / N + 1e-9)
        return [es, ls, supp, rank, len(pr["cands"]), pr["n_subj"] / max(pr["nrows"], 1),
                float(pr["obj_ent"]), hco, hcs, pr_log, float(best_subj.get(pr["tid"]) == pr["i"])]

    folds = frozen_kfold(sorted({r.table_id for r in recs}), k=5, seed=0)

    def run(drop):
        hit = tot = 0
        for test in folds:
            ts = set(test)
            prior = Counter()
            for pr in pairs:
                if pr["tid"] not in ts and pr["gold"]:
                    prior[pr["gold"]] += 1
            N = sum(prior.values()) or 1
            X, y = [], []
            for pr in pairs:
                if pr["tid"] in ts:
                    continue
                for p in pr["cands"]:
                    X.append(feats(pr, p, prior, N))
                    y.append(1 if p == pr["gold"] else 0)
            Xa = np.array(X)
            if drop is not None:
                Xa = np.delete(Xa, drop, axis=1)
            clf = HistGradientBoostingClassifier(max_iter=400, max_depth=4, learning_rate=0.06,
                                                 min_samples_leaf=30, l2_regularization=1.0).fit(Xa, np.array(y))
            for pr in pairs:
                if pr["tid"] not in ts or not pr["gold"] or pr["gold"] not in pr["cands"]:
                    continue
                tot += 1  # reachable gold pair
                xs = np.array([feats(pr, p, prior, N) for p in pr["cands"]])
                if drop is not None:
                    xs = np.delete(xs, drop, axis=1)
                pred = pr["cands"][int(clf.predict_proba(xs)[:, 1].argmax())]
                if pred == pr["gold"]:
                    hit += 1
        return hit / tot if tot else 0.0

    full = run(None)
    print("=" * 56)
    print("CPA feature ablation (250WT, gold, seed0, argmax-recall on reachable)")
    print(f"  FULL model argmax-recall: {full:.3f}")
    print("-" * 56)
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        f = run(i)
        rows.append((name, f, full - f))
        print(f"  -{name:18s}: {f:.3f}  (drop {full - f:+.3f})")
    rows.sort(key=lambda x: -x[2])
    print("-" * 56)
    print("  most important:", ", ".join(n for n, _, _ in rows[:3]))
    print("=" * 56)


if __name__ == "__main__":
    main()
