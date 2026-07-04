"""LLM baseline on ToughTables (2T_WD) CTA, official AH scorer, candgen top-1.

Mirrors flint_toughtables.py's candidate construction (real candgen entities,
ancestor-closure type candidates) but the SELECTOR is an LLM (constrained choice
over the same candidates). Comparable to: FLINT in-domain 0.616, counting 0.565,
published DAGOBAH 0.409 / KGCODE 0.543. Partial-cache lower bound (same caveat).

Usage: python scripts/llm_toughtables.py --model gpt-4o
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

from flint.data.candgen import CachedCandidateGenerator
from flint.data.kg import CachedWikidataKG, ancestors_with_hops
from flint.data.loaders import load_dataset
from flint.eval.toughtables_score import score_official
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_baseline import api_generator, load_llm, chat_batch, pick_qid, MAX_CANDS, MAXROWS

GT_DIR = Path("data/raw/zenodo/2T_WD/2T_WD/gt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()
    kg = CachedWikidataKG(offline=True)
    cg = CachedCandidateGenerator(offline=True)
    lab = lambda q: (kg.label(q) or q)  # noqa: E731
    if args.model == "qwen":
        tok, model = load_llm()
        gen_fn = lambda ps: chat_batch(tok, model, ps)  # noqa: E731
    else:
        gen_fn = api_generator(args.model)

    recs = list(load_dataset("toughtables"))
    prompts, keys, valids = [], [], []
    ncols = 0
    for r in recs:
        nr = min(len(r.rows), MAXROWS)
        for col in r.entity_columns:
            ncols += 1
            ents = []
            for row in range(nr):
                cell = r.rows[row][col] if col < len(r.rows[row]) else ""
                c = cg.candidates(cell, k=1) if cell.strip() else []
                if c:
                    ents.append(c[0])
            if not ents:
                continue
            clos = Counter(t for e in ents for d in kg.instance_of(e)
                           for t in ancestors_with_hops(kg, d, max_hops=4))
            if not clos:
                continue
            cands = [q for q, _ in clos.most_common(MAX_CANDS)]
            header = r.headers[col] if col < len(r.headers) else ""
            cells = [r.rows[row][col] for row in range(min(8, len(r.rows))) if col < len(r.rows[row])]
            opts = "\n".join(f"{q}: {lab(q)}" for q in cands)
            p = (f"A table column (header: \"{header}\") contains these values:\n"
                 f"{', '.join(cells)}\n\nWhich ONE Wikidata type best describes this column? "
                 f"Choose from:\n{opts}\n\nAnswer with only the Q-id.")
            prompts.append(p); keys.append((r.table_id, col)); valids.append(set(cands) | {cands[0]})
    print(f"[ToughTables CTA] {len(prompts)}/{ncols} columns ({len(prompts)/max(ncols,1):.1%} cov)", flush=True)
    t0 = time.time()
    outs = gen_fn(prompts)
    pred = {}
    for (tid, col), text, valid in zip(keys, outs, valids):
        q = pick_qid(text, valid) or next(iter(valid))
        pred.setdefault(tid, {})[col] = q
    m = score_official(pred, GT_DIR)
    print("=" * 64)
    print(f"ToughTables CTA | LLM ({args.model}) candgen top-1 | OFFICIAL AH ({time.time()-t0:.0f}s)")
    print(f"  [ref] DAGOBAH 0.409 | KGCODE 0.543 | counting 0.565 | FLINT in-domain 0.616")
    print(f"  LLM-CTA: AH={m['AH']:.3f}  AP={m['AP']:.3f}  F1={m['F1']:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
