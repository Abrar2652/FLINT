"""FLINT-CTA: a lightweight learned ranker for column-type annotation.

Thesis under test: GRAMS+'s CTA accuracy (ancestor-climbing + trained MLP) can be
matched or beaten by a *tiny* feature-based ranker over closed-form ontology
signals -- no GNN, no Steiner tree, no message passing. The ranker scores each
candidate type t in a column's ancestor-closure with a HistGradientBoosting
regressor trained to predict t's cscore against gold, then predicts argmax.

Features per (column, candidate-type t):
  - coverage(t): fraction of column cells whose P31 ancestor-closure contains t
  - coverage rank within the column
  - is_mode: t is the majority direct (P31) type
  - is_anc_of_mode / hops_above_mode: t generalises the mode by k hops
  - n_ancestors(t): generality / depth proxy
  - header<->label cosine (SBERT), and column-context<->label cosine
  - log gold-frequency prior of t (estimated on TRAIN folds only)

Eval: SemTab approximate hierarchical cscore (micro-F1), 5-fold CV on 250WT,
gold entities. Baselines printed for reference (counting 0.722, GRAMS+ 0.789).
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor

from flint.data.candgen import CachedCandidateGenerator
from flint.data.kg import CachedWikidataKG, ancestors_with_hops, hierarchy_distance, is_ancestor
from flint.data.loaders import load_dataset
from flint.data.splits import frozen_kfold
from flint.eval.evaluate import evaluate_cta
from flint.eval.metrics import cscore
from flint.graph.features import SentenceTransformerEncoder

MAX_HOPS = 4
COV_FLOOR = 0.2
TYPE_PROPS = {"P106", "P641", "P39", "P101", "P452", "P136", "P413", "P39"}


def linked_entities(r, col, entity_source, cg):
    """Per-row entity for a column: gold CEA (oracle) or candgen top-1 (real EL).
    Returns list of entity ids (one per row that resolves)."""
    out = []
    for row in range(len(r.rows)):
        if entity_source == "gold":
            e = r.gold_cea.get((row, col))
        else:
            cell = r.rows[row][col] if col < len(r.rows[row]) else ""
            cands = cg.candidates(cell, k=1) if cell.strip() else []
            e = cands[0] if cands else None
        if e:
            out.append(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="250wt", help="250wt|hardtables|wikidatatables|...")
    ap.add_argument("--entities", choices=["gold", "candgen"], default="gold",
                    help="gold = oracle CEA (identical protocol); candgen = real top-1 EL")
    args = ap.parse_args()
    kg = CachedWikidataKG(offline=True)
    cg = CachedCandidateGenerator(offline=True) if args.entities == "candgen" else None
    recs = list(load_dataset(args.dataset))
    gold = {r.table_id: r.gold_cta for r in recs}
    enc = SentenceTransformerEncoder()
    n_cols_total = sum(len(r.entity_columns) for r in recs)

    def lab(q: str) -> str:
        try:
            return kg.label(q) or q
        except Exception:  # noqa: BLE001
            return q

    # ---- per-column candidate sets + raw signals ----
    cols = []  # dict per column
    labset: set[str] = set()
    entset: set[str] = set()
    for r in recs:
        for col in r.entity_columns:
            ents = linked_entities(r, col, args.entities, cg)
            if not ents:
                continue
            clos = [{t for d in kg.instance_of(e) for t in ancestors_with_hops(kg, d, max_hops=MAX_HOPS)}
                    for e in ents]
            clos = [c for c in clos if c]
            n = len(clos) or 1
            cnt = Counter(t for c in clos for t in c)
            cov = {t: cnt[t] / n for t in cnt}
            # direct-type coverage: fraction of cells whose P31 directly == t
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
            # add type-conferring property values + their closure as extra candidates
            for e in ents:
                for p, o in kg.statements(e):
                    if p in TYPE_PROPS and o.startswith("Q"):
                        cands.update(ancestors_with_hops(kg, o, max_hops=2))
            cands = list(cands) or [mode]
            header = r.headers[col] if col < len(r.headers) else ""
            sample_cells = [r.rows[row][col] for row in range(min(8, len(r.rows)))
                            if col < len(r.rows[row])]
            ctx = (header + " : " + ", ".join(sample_cells))[:256]
            ent_sample = list(dict.fromkeys(ents))[:10]
            cols.append(dict(tid=r.table_id, col=col, cov=cov, dcov=dcov, mode=mode,
                             cands=cands, header=header, ctx=ctx, ents=ent_sample))
            labset.update(cands)
            entset.update(ent_sample)

    if args.entities == "candgen":
        print(f"[real-candidate protocol] columns with >=1 linked cell: "
              f"{len(cols)}/{n_cols_total} ({len(cols)/max(n_cols_total,1):.1%} coverage)")

    # ---- SBERT embeddings (precompute once) ----
    labs = sorted(labset)
    L2i = {q: i for i, q in enumerate(labs)}
    lab_emb = F.normalize(enc.encode([lab(q) for q in labs]), dim=1)
    hdrs = sorted({c["header"] for c in cols})
    H2i = {h: i for i, h in enumerate(hdrs)}
    hdr_emb = F.normalize(enc.encode(hdrs), dim=1)
    ctxs = sorted({c["ctx"] for c in cols})
    C2i = {c: i for i, c in enumerate(ctxs)}
    ctx_emb = F.normalize(enc.encode(ctxs), dim=1)
    ents_l = sorted(entset)
    E2i = {e: i for i, e in enumerate(ents_l)}
    ent_emb = F.normalize(enc.encode([lab(e) for e in ents_l]), dim=1)
    # per-column mean entity-label embedding
    for ci in cols:
        idx = [E2i[e] for e in ci["ents"] if e in E2i]
        ci["ementb"] = F.normalize(ent_emb[idx].mean(0), dim=0) if idx else None

    n_anc_cache: dict[str, int] = {}

    def n_anc(t: str) -> int:
        if t not in n_anc_cache:
            n_anc_cache[t] = len(ancestors_with_hops(kg, t, max_hops=6))
        return n_anc_cache[t]

    def feats(ci, t, prior, N, cond):
        cov, dcov, mode, cands = ci["cov"], ci["dcov"], ci["mode"], ci["cands"]
        c = cov.get(t, 0.0)
        rank = sum(1 for u in cands if cov.get(u, 0) > c) / max(len(cands), 1)
        is_mode = float(t == mode)
        anc_of_mode = float(is_ancestor(kg, t, mode))
        d = hierarchy_distance(kg, t, mode) if anc_of_mode else 0
        hcos = float(hdr_emb[H2i[ci["header"]]] @ lab_emb[L2i[t]])
        xcos = float(ctx_emb[C2i[ci["ctx"]]] @ lab_emb[L2i[t]])
        ecos = float(ci["ementb"] @ lab_emb[L2i[t]]) if ci["ementb"] is not None else 0.0
        pr = math.log(prior.get(t, 0) / N + 1e-9)
        return [c, dcov.get(t, 0.0), rank, is_mode, anc_of_mode, float(d or 0),
                float(n_anc(t)), hcos, xcos, ecos, pr]

    isa = lambda a, b: is_ancestor(kg, a, b)  # noqa: E731
    hd = lambda a, b: hierarchy_distance(kg, a, b)  # noqa: E731

    def fit_predict(fold_seed):
        folds = frozen_kfold(sorted({r.table_id for r in recs}), k=5, seed=fold_seed, name=args.dataset)
        pred = {r.table_id: {} for r in recs}
        for test in folds:
            testset = set(test)
            prior = Counter()
            for r in recs:
                if r.table_id in testset:
                    continue
                for g in r.gold_cta.values():
                    if g:
                        prior[g] += 1
            N = sum(prior.values()) or 1
            X, yc, ye = [], [], []
            for ci in cols:
                if ci["tid"] in testset:
                    continue
                g = gold[ci["tid"]].get(ci["col"])
                if not g:
                    continue
                for t in ci["cands"]:
                    X.append(feats(ci, t, prior, N, None))
                    yc.append(cscore(t, g, hierarchy_distance=hd, is_ancestor=isa))
                    ye.append(1.0 if t == g else 0.0)
            Xa = np.array(X)
            # blend a partial-credit model (cscore geometry) with an exact-match model
            # (sharpens granularity) -- both lightweight, no NN.
            reg_c = HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.06,
                                                  min_samples_leaf=30, l2_regularization=1.0).fit(Xa, np.array(yc))
            reg_e = HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.06,
                                                  min_samples_leaf=30, l2_regularization=1.0).fit(Xa, np.array(ye))
            for ci in cols:
                if ci["tid"] not in testset:
                    continue
                cands = ci["cands"]
                xs = np.array([feats(ci, t, prior, N, None) for t in cands])
                scores = reg_c.predict(xs) + reg_e.predict(xs)
                pred[ci["tid"]][ci["col"]] = cands[int(scores.argmax())]
        return pred

    # ---- deterministic baselines on the IDENTICAL protocol ----
    count_pred, grams_pred = {}, {}
    for ci in cols:
        tid, col, cov, mode = ci["tid"], ci["col"], ci["cov"], ci["mode"]
        count_pred.setdefault(tid, {})[col] = mode
        cur, delta = mode, 0.10
        for _ in range(2):  # GRAMS+ CTA: ancestor-climb, max_distance=2
            anc = [a for a in ancestors_with_hops(kg, cur, max_hops=1) if a != cur and a in cov]
            if not anc:
                break
            best = max(anc, key=lambda a: cov[a])
            if cov[best] >= cov.get(cur, 0) + delta:
                cur = best
            else:
                break
        grams_pred.setdefault(tid, {})[col] = cur

    # ---- run FLINT over 5 fold-seeds (CV-shuffle variance) ----
    seeds = [0, 1, 2, 3, 4]
    flint_preds = [fit_predict(s) for s in seeds]
    # ---- dump per-table CTA F1 (seed 0) for paired significance vs real SOTA ----
    import json as _json, os as _os
    _pt = {}
    for r in recs:
        gc = gold[r.table_id]
        if not gc:
            continue
        # per-table CTA F1 = sum cscore over predicted gold cols / #GOLD cols (missed=0),
        # so it counts FLINT's recall misses exactly like the artifact systems' per-table F1.
        pr0 = flint_preds[0].get(r.table_id, {})
        ssum = sum(cscore(pr0[c], g, hierarchy_distance=hd, is_ancestor=isa)
                   for c, g in gc.items() if c in pr0)
        _pt[r.table_id] = {"f": ssum / len(gc), "hdr": "|".join(map(str, r.headers))}
    _os.makedirs("experiments/flint", exist_ok=True)
    open(f"experiments/flint/pertable_cta_{args.dataset}.json", "w").write(_json.dumps(_pt))
    flint_evals = [evaluate_cta(p, gold, kg) for p in flint_preds]
    flint_micro = [e["cta_micro_f1"] for e in flint_evals]
    import statistics
    mc = evaluate_cta(count_pred, gold, kg)["cta_micro_f1"]
    mg = evaluate_cta(grams_pred, gold, kg)["cta_micro_f1"]
    fmean, fstd = statistics.mean(flint_micro), statistics.stdev(flint_micro)

    # ---- full m-P / m-R / m-F breakdown (GRAMS+ paper format), micro & macro ----
    def avg(key):
        return statistics.mean(e[key] for e in flint_evals)
    cnt_e = evaluate_cta(count_pred, gold, kg)
    grm_e = evaluate_cta(grams_pred, gold, kg)
    print("=" * 64)
    print(f"{args.dataset} CTA | {('gold entities' if args.entities=='gold' else 'REAL EL')} | m-P / m-R / m-F")
    print(f"  {'method':22s} {'micro-P':>8} {'micro-R':>8} {'micro-F':>8} | {'macro-P':>8} {'macro-R':>8} {'macro-F':>8}")
    for nm, e in [("counting", cnt_e), ("GRAMS+ algorithm", grm_e)]:
        print(f"  {nm:22s} {e['cta_micro_p']:8.3f} {e['cta_micro_r']:8.3f} {e['cta_micro_f1']:8.3f} | "
              f"{e['cta_macro_p']:8.3f} {e['cta_macro_r']:8.3f} {e['cta_macro_f1']:8.3f}")
    print(f"  {'FLINT (ours)':22s} {avg('cta_micro_p'):8.3f} {avg('cta_micro_r'):8.3f} {avg('cta_micro_f1'):8.3f} | "
          f"{avg('cta_macro_p'):8.3f} {avg('cta_macro_r'):8.3f} {avg('cta_macro_f1'):8.3f}")

    # ---- paired sign test: FLINT (seed 0) vs GRAMS+ algorithm, per column ----
    wins = losses = 0
    for ci in cols:
        tid, col = ci["tid"], ci["col"]
        g = gold[tid].get(col)
        if not g:
            continue
        fs = cscore(flint_preds[0][tid][col], g, hierarchy_distance=hd, is_ancestor=isa)
        gs = cscore(grams_pred[tid][col], g, hierarchy_distance=hd, is_ancestor=isa)
        if fs > gs + 1e-9:
            wins += 1
        elif gs > fs + 1e-9:
            losses += 1
    n = wins + losses
    # two-sided sign test p-value via normal approx
    from math import erf, sqrt
    z = (wins - n / 2) / (sqrt(n) / 2) if n else 0.0
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

    proto = "gold entities" if args.entities == "gold" else "REAL candgen top-1 EL"
    # published CTA reference per dataset (protocol/metric noted; reference only)
    REF = {"250wt": "GRAMS+ P2 0.789 (m-AF)", "hardtables": "MTab 0.915 / GRAMS+ 0.906 (m-AF)",
           "wikidatatables": "MTab 0.941 / GRAMS+ 0.941 (m-AF)"}
    print("=" * 64)
    print(f"{args.dataset} CTA | {proto} | identical candidate gen | micro cscore")
    print("-" * 64)
    print(f"  counting (majority direct P31)          : {mc:.3f}")
    print(f"  GRAMS+ algorithm (ancestor-climb d=.1)  : {mg:.3f}")
    print(f"  FLINT-CTA learned ranker (ours)         : {fmean:.3f} +/- {fstd:.3f}  (5 fold-seeds)")
    print(f"  [ref] published (diff protocol): {REF.get(args.dataset, 'n/a')}")
    print("-" * 64)
    print(f"  per-column sign test FLINT vs GRAMS+ algo: "
          f"{wins} wins / {losses} losses (n={n}), p={pval:.2e}")
    print("=" * 64)


if __name__ == "__main__":
    main()
