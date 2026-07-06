"""Paired significance test: FLINT vs the ACTUAL GRAMS+, MTab, DAGOBAH on 250WT.

Uses the GRAMS+ released artifact, which stores per-table cta-f/cpa-f for all three
systems (individual/NNN/primitive). FLINT per-table F1 comes from the dumps written by
flint_cta_ranker / flint_cpa_ranker (experiments/flint/pertable_{cta,cpa}_250wt.json).
Alignment: artifact index order == sm Dataset load order (verified by header signature).

Reports, for CTA and CPA: mean per-table F1, paired sign test (wins/losses, p), and a
paired bootstrap CI on the mean difference. This is the field-standard system-vs-system
test GRAMS+ itself used (its MTab sign test was p=0.086). Caveat: FLINT is gold-EL,
published rows are own-EL — a system-level comparison, not EL-controlled.
"""
from __future__ import annotations

import json
import re
from math import erf, sqrt
from pathlib import Path

import h5py

from flint.data.loaders import load_dataset

ART = Path("/tmp/iswc24-gp/experiments/overall-performance")
SYS = {"GRAMS+": "our250wt", "MTab": "mtab250wt", "DAGOBAH": "dagobah250wt"}


def artifact_pertable(task):
    """index(int) -> {system: f1}, plus index -> headers signature (from GRAMS+ table)."""
    scores = {}
    headers = {}
    for name, d in SYS.items():
        with h5py.File(ART / d / "data.h5", "r") as h:
            keys = sorted(h["individual"].keys())
            for i, k in enumerate(keys):
                g = h["individual"][k]["primitive"]
                scores.setdefault(i, {})[name] = float(g[f"{task}-f"][()])
                if name == "GRAMS+" and i not in headers:
                    try:
                        tj = json.loads(h["individual"][k]["complex"]["table"][()])
                        hdr = [re.sub(r"^\d+\.\s*", "", c) for c in tj["rows"][0].keys()]
                        headers[i] = "|".join(hdr)
                    except Exception:  # noqa: BLE001
                        headers[i] = None
    return scores, headers


def sign_test(diffs):
    w = sum(1 for d in diffs if d > 1e-9)
    l = sum(1 for d in diffs if d < -1e-9)
    n = w + l
    if not n:
        return w, l, 1.0
    z = (w - n / 2) / (sqrt(n) / 2)
    return w, l, 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def bootstrap(diffs, B=5000):
    import random
    rng = random.Random(0)
    n = len(diffs)
    means = []
    for _ in range(B):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * B)], means[int(0.975 * B)]


def run(task, flint_file=None, proto="oracle-EL"):
    fpath = Path(flint_file) if flint_file else Path(f"experiments/flint/pertable_{task}_250wt.json")
    flint = json.loads(fpath.read_text())
    recs = list(load_dataset("250wt"))
    art, art_hdr = artifact_pertable(task)
    # align by load-order index; verify header signature agreement rate
    agree = tot = 0
    paired = {s: [] for s in SYS}      # per system: list of (flint_f, sys_f)
    flint_means = []
    for i, r in enumerate(recs):
        tid = r.table_id
        if tid not in flint or i not in art:
            continue
        fhdr = "|".join(map(str, r.headers))
        if art_hdr.get(i):
            tot += 1
            agree += (fhdr.lower() == art_hdr[i].lower())
        ff = flint[tid]["f"]
        flint_means.append(ff)
        for s in SYS:
            paired[s].append((ff, art[i][s]))
    print(f"\n=== {task.upper()} [{proto}] (250WT, per-table, n={len(flint_means)}) ===")
    print(f"  header-alignment check: {agree}/{tot} match ({agree/max(tot,1):.0%})")
    print(f"  FLINT mean per-table F1: {sum(flint_means)/len(flint_means):.4f}")
    for s in SYS:
        ff = [a for a, _ in paired[s]]
        sf = [b for _, b in paired[s]]
        diffs = [a - b for a, b in paired[s]]
        w, l, p = sign_test(diffs)
        lo, hi = bootstrap(diffs)
        sysmean = sum(sf) / len(sf)
        md = sum(diffs) / len(diffs)
        verdict = ("FLINT wins" if md > 0 and p < 0.05 else
                   "FLINT loses" if md < 0 and p < 0.05 else "n.s. (tie)")
        print(f"  vs {s:8s}: {s} mean={sysmean:.4f} | dFLINT={md:+.4f} "
              f"[{lo:+.3f},{hi:+.3f}] | sign {w}W/{l}L p={p:.2e} -> {verdict}")


def main():
    import sys
    if "--real-el" in sys.argv:
        # W1 fix: paired significance + bootstrap CI on the standard end-to-end (each
        # system its OWN EL) P1 comparison -- FLINT real top-1 EL vs GRAMS+/MTab/DAGOBAH.
        # This tests whether the 0.654-vs-0.664 CPA gap is significant (reviewer W1).
        run("cpa", "experiments/flint/pertable_cpa_250wt_candgen.json",
            "real-EL P1: FLINT top-1 EL vs each system own-EL")
    else:
        for task in ["cta", "cpa"]:
            run(task)


if __name__ == "__main__":
    main()
