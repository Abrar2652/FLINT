"""FLINT-CTA cross-dataset generalization on ToughTables (2T_WD).

Zero-shot transfer: train the FLINT-CTA ranker on 250WT (gold entities), apply it
ZERO-SHOT to ToughTables columns built from real candgen top-1 entities. Compare
counting / GRAMS+-algorithm / FLINT under the OFFICIAL SemTab AH scorer (so numbers
are comparable to published DAGOBAH 0.409 / KGCODE 0.543). Honest caveat: the
candidate cache is partial (warming, rate-limited), so AH is understated for ALL
methods by missing-candidate columns (scored 0); the comparison across methods is
on identical candidates and therefore fair.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor

from flint.data.candgen import CachedCandidateGenerator
from flint.data.kg import CachedWikidataKG, ancestors_with_hops, hierarchy_distance, is_ancestor
from flint.data.loaders import load_dataset
from flint.eval.metrics import cscore
from flint.eval.toughtables_score import score_official
from flint.graph.features import SentenceTransformerEncoder

MAX_HOPS = 4
COV_FLOOR = 0.2
TYPE_PROPS = {"P106", "P641", "P39", "P101", "P452", "P136", "P413"}
GT_DIR = Path("data/raw/zenodo/2T_WD/2T_WD/gt")
MAXROWS = 20


def build_columns(recs, kg, entity_fn):
    """One dict per entity column: coverage map, mode, candidate types, text ctx,
    linked entities. entity_fn(r,row,col) -> entity id or None."""
    cols = []
    for r in recs:
        nr = min(len(r.rows), MAXROWS)
        for col in r.entity_columns:
            ents = [e for row in range(nr) if (e := entity_fn(r, row, col))]
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
            sample = [r.rows[row][col] for row in range(nr) if col < len(r.rows[row])][:8]
            ctx = (header + " : " + ", ".join(sample))[:256]
            es = list(dict.fromkeys(ents))[:10]
            cols.append(dict(tid=r.table_id, col=col, cov=cov, dcov=dcov, mode=mode,
                             cands=cands, header=header, ctx=ctx, ents=es))
    return cols


def main() -> None:
    t0 = time.time()
    kg = CachedWikidataKG(offline=True)
    cg = CachedCandidateGenerator(offline=True)
    enc = SentenceTransformerEncoder()
    lab = lambda q: (kg.label(q) or q)  # noqa: E731

    wt = list(load_dataset("250wt"))
    tt = list(load_dataset("toughtables"))
    wt_gold = {r.table_id: r.gold_cta for r in wt}

    gold_fn = lambda r, row, col: r.gold_cea.get((row, col))  # noqa: E731

    def cand_fn(r, row, col):
        cell = r.rows[row][col] if col < len(r.rows[row]) else ""
        c = cg.candidates(cell, k=1) if cell.strip() else []
        return c[0] if c else None

    wt_cols = build_columns(wt, kg, gold_fn)            # train (250WT gold)
    tt_cols = build_columns(tt, kg, cand_fn)            # test (ToughTables candgen)
    n_tt_targets = sum(len(r.entity_columns) for r in tt)
    print(f"250WT train cols {len(wt_cols)} | ToughTables test cols {len(tt_cols)}/"
          f"{n_tt_targets} ({len(tt_cols)/max(n_tt_targets,1):.1%} coverage) | {time.time()-t0:.0f}s",
          flush=True)

    # ---- shared SBERT space ----
    labset = {t for c in wt_cols + tt_cols for t in c["cands"]}
    labs = sorted(labset); L = {q: i for i, q in enumerate(labs)}
    lab_emb = F.normalize(enc.encode([lab(q) for q in labs]), dim=1)
    hdrs = sorted({c["header"] for c in wt_cols + tt_cols}); H = {h: i for i, h in enumerate(hdrs)}
    hdr_emb = F.normalize(enc.encode(hdrs), dim=1)
    ctxs = sorted({c["ctx"] for c in wt_cols + tt_cols}); C = {c: i for i, c in enumerate(ctxs)}
    ctx_emb = F.normalize(enc.encode(ctxs), dim=1)
    entset = {e for c in wt_cols + tt_cols for e in c["ents"]}
    el = sorted(entset); E = {e: i for i, e in enumerate(el)}
    ent_emb = F.normalize(enc.encode([lab(e) for e in el]), dim=1)
    for ci in wt_cols + tt_cols:
        idx = [E[e] for e in ci["ents"] if e in E]
        ci["em"] = F.normalize(ent_emb[idx].mean(0), dim=0) if idx else None

    nac: dict = {}
    def n_anc(t):
        if t not in nac:
            nac[t] = len(ancestors_with_hops(kg, t, max_hops=6))
        return nac[t]

    # gold-frequency prior from 250WT gold types (transfers to ToughTables)
    prior = Counter()
    for r in wt:
        for g in r.gold_cta.values():
            if g:
                prior[g] += 1
    N = sum(prior.values()) or 1

    def feats(ci, t):
        cov, dcov, mode, cands = ci["cov"], ci["dcov"], ci["mode"], ci["cands"]
        c = cov.get(t, 0.0)
        rank = sum(1 for u in cands if cov.get(u, 0) > c) / max(len(cands), 1)
        anc = float(is_ancestor(kg, t, mode))
        d = hierarchy_distance(kg, t, mode) if anc else 0
        hco = float(hdr_emb[H[ci["header"]]] @ lab_emb[L[t]])
        xco = float(ctx_emb[C[ci["ctx"]]] @ lab_emb[L[t]])
        eco = float(ci["em"] @ lab_emb[L[t]]) if ci["em"] is not None else 0.0
        return [c, dcov.get(t, 0.0), rank, float(t == mode), anc, float(d or 0),
                float(n_anc(t)), hco, xco, eco, math.log(prior.get(t, 0) / N + 1e-9)]

    isa = lambda a, b: is_ancestor(kg, a, b)  # noqa: E731
    hd = lambda a, b: hierarchy_distance(kg, a, b)  # noqa: E731

    # ---- train FLINT ranker on 250WT (blended cscore + exact) ----
    X, yc, ye = [], [], []
    for ci in wt_cols:
        g = wt_gold[ci["tid"]].get(ci["col"])
        if not g:
            continue
        for t in ci["cands"]:
            X.append(feats(ci, t))
            yc.append(cscore(t, g, hierarchy_distance=hd, is_ancestor=isa))
            ye.append(1.0 if t == g else 0.0)
    Xa = np.array(X)
    rc = HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.06,
                                       min_samples_leaf=30, l2_regularization=1.0).fit(Xa, np.array(yc))
    re = HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.06,
                                       min_samples_leaf=30, l2_regularization=1.0).fit(Xa, np.array(ye))

    # ---- predict ToughTables: counting / GRAMS+-algo / FLINT ----
    cnt_pred, grm_pred, fl_pred = {}, {}, {}
    for ci in tt_cols:
        tid, col, cov, mode, cands = ci["tid"], ci["col"], ci["cov"], ci["mode"], ci["cands"]
        cnt_pred.setdefault(tid, {})[col] = mode
        cur = mode
        for _ in range(2):
            anc = [a for a in ancestors_with_hops(kg, cur, max_hops=1) if a != cur and a in cov]
            if not anc:
                break
            best = max(anc, key=lambda a: cov[a])
            if cov[best] >= cov.get(cur, 0) + 0.10:
                cur = best
            else:
                break
        grm_pred.setdefault(tid, {})[col] = cur
        xs = np.array([feats(ci, t) for t in cands])
        s = rc.predict(xs) + re.predict(xs)
        fl_pred.setdefault(tid, {})[col] = cands[int(s.argmax())]

    # ---- FLINT with IN-DOMAIN training: 5-fold CV within ToughTables ----
    # multigold target: max cscore over the column's accept set.
    accept = {}
    for r in tt:
        for c, s in r.gold_cta_accept.items():
            accept[(r.table_id, c)] = set(s)
    tt_tids = sorted({ci["tid"] for ci in tt_cols})
    folds = [set(tt_tids[k::5]) for k in range(5)]
    flcv_pred: dict = {}
    for fv in folds:
        Xt, yt = [], []
        for ci in tt_cols:
            if ci["tid"] in fv:
                continue
            acc = accept.get((ci["tid"], ci["col"]))
            if not acc:
                continue
            for t in ci["cands"]:
                Xt.append(feats(ci, t))
                yt.append(max(cscore(t, g, hierarchy_distance=hd, is_ancestor=isa) for g in acc))
        if not Xt:
            continue
        reg = HistGradientBoostingRegressor(max_iter=400, max_depth=4, learning_rate=0.06,
                                            min_samples_leaf=30, l2_regularization=1.0).fit(np.array(Xt), np.array(yt))
        for ci in tt_cols:
            if ci["tid"] not in fv:
                continue
            xs = np.array([feats(ci, t) for t in ci["cands"]])
            flcv_pred.setdefault(ci["tid"], {})[ci["col"]] = ci["cands"][int(reg.predict(xs).argmax())]

    cnt = score_official(cnt_pred, GT_DIR)
    grm = score_official(grm_pred, GT_DIR)
    fl = score_official(fl_pred, GT_DIR)
    flcv = score_official(flcv_pred, GT_DIR)
    print("=" * 66)
    print("ToughTables (2T_WD) CTA | real candgen top-1 | OFFICIAL AH scorer")
    print("-" * 66)
    print(f"  [ref] published SOTA: DAGOBAH AH 0.409 | KGCODE AH 0.543")
    print(f"  counting (majority P31)        : AH={cnt['AH']:.3f}  AP={cnt['AP']:.3f}  F1={cnt['F1']:.3f}")
    print(f"  GRAMS+ algorithm (ancestor-climb): AH={grm['AH']:.3f}  AP={grm['AP']:.3f}  F1={grm['F1']:.3f}")
    print(f"  FLINT-CTA zero-shot from 250WT : AH={fl['AH']:.3f}  AP={fl['AP']:.3f}  F1={fl['F1']:.3f}")
    print(f"  FLINT-CTA in-domain 5-fold CV  : AH={flcv['AH']:.3f}  AP={flcv['AP']:.3f}  F1={flcv['F1']:.3f}")
    print(f"  (targets={fl['n_targets']} predicted={fl['n_predicted']}; partial cache understates AH)")
    print("=" * 66)


if __name__ == "__main__":
    main()
