"""LLM baseline for STI (CTA + CPA), constrained-choice over IDENTICAL candidates.

Fair head-to-head vs FLINT: the LLM is given the SAME candidate set FLINT's ranker
scores (the column's ancestor-closure types for CTA; the per-pair candidate
properties for CPA) and must pick one (or NONE for CPA). This isolates SELECTION
quality: a tiny gradient-boosted ranker vs a 7B instruct LLM, same inputs, same
metric (CTA approximate cscore; CPA exact micro-F1). Local open model (no API),
bf16 on one GPU. Gold entities (identical protocol).

Usage:
  python scripts/llm_baseline.py --task cta --dataset 250wt
  python scripts/llm_baseline.py --task cpa --dataset 250wt
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import torch

from flint.data.kg import (CachedWikidataKG, ancestors_with_hops, hierarchy_distance,
                           is_ancestor, literal_matches)
from flint.data.loaders import load_dataset
from flint.eval.evaluate import evaluate_cta, evaluate_cpa
from flint.utils.paths import cache_dir

QWEN = "Qwen/Qwen2.5-7B-Instruct"
MAX_CANDS = 30
MAXROWS = 20

# cluster-local open models: short-name -> (hf_repo_id, cache_dir)
LOCAL_MODELS = {
    "qwen25-7b": ("Qwen/Qwen2.5-7B-Instruct", "/nas/ckgfs/jaunts/jahin/hf_cache/hub"),
    "qwen25-3b": ("Qwen/Qwen2.5-3B-Instruct", "/nas/ckgfs/users/jahin/hf_cache/hub"),
    "qwen-14b": ("Qwen/Qwen-14B-Chat", "/nas/ckgfs/users/jahin/hf_cache/hub"),
    "llama31-8b": ("meta-llama/Llama-3.1-8B-Instruct", "/nas/ckgfs/users/jahin/hf_cache/hub"),
    "gemma2-9b": ("google/gemma-2-9b-it", "/nas/ckgfs/users/jahin/hf_cache/hub"),
    "ministral-8b": ("mistralai/Ministral-8B-Instruct-2410", "/nas/ckgfs/users/jahin/hf_cache/hub"),
    "mistral-7b": ("mistralai/Mistral-7B-Instruct-v0.2", "/nas/ckgfs/jaunts/jahin/.hf_cache"),
    "phi3-medium": ("microsoft/Phi-3-medium-4k-instruct", "/nas/ckgfs/jaunts/jahin/.hf_cache"),
    "gptoss-20b": ("openai/gpt-oss-20b", "/nas/ckgfs/users/jahin/hf_cache/hub"),
    "deepseek-r1-7b": ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "/nas/ckgfs/users/jahin/hf_cache/hub"),
}


def api_generator(model_name):
    """Return gen_fn(prompts)->list[str] backed by a frontier API (gpt*/claude*).
    Key loaded from flint/.env at runtime only (never logged)."""
    from concurrent.futures import ThreadPoolExecutor
    from dotenv import dotenv_values
    env = dotenv_values("/nas/ckgfs/jaunts/jahin/flint/.env")
    provider = ("claude" if model_name.startswith("claude")
                else "gemini" if model_name.startswith("gemini") else "openai")
    if provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=env["ANTHROPIC_API_KEY"])
    elif provider == "gemini":
        from google import genai
        client = genai.Client(api_key=env["GOOGLE_API_KEY"])
    else:
        import openai
        client = openai.OpenAI(api_key=env["OPENAI_API_KEY"])

    def one(p):
        for attempt in range(5):
            try:
                if provider == "claude":
                    r = client.messages.create(model=model_name, max_tokens=12,
                                               messages=[{"role": "user", "content": p}])
                    return r.content[0].text
                if provider == "gemini":
                    from google.genai import types
                    cfg = types.GenerateContentConfig(
                        max_output_tokens=12,
                        thinking_config=types.ThinkingConfig(thinking_budget=0))
                    r = client.models.generate_content(model=model_name, contents=p, config=cfg)
                    return r.text or ""
                r = client.chat.completions.create(model=model_name, max_tokens=12,
                                                   temperature=0,
                                                   messages=[{"role": "user", "content": p}])
                return r.choices[0].message.content or ""
            except Exception:  # noqa: BLE001
                import time
                time.sleep(min(2 ** attempt, 20))
        return ""

    def gen_fn(prompts):
        out = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i, r in enumerate(ex.map(one, prompts)):
                out[i] = r
        return out

    return gen_fn


def load_llm(short="qwen25-7b"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, cache = LOCAL_MODELS[short]
    tok = AutoTokenizer.from_pretrained(repo, cache_dir=cache, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, cache_dir=cache, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok, model


@torch.no_grad()
def chat_batch(tok, model, prompts, max_new=48):
    """Batched greedy generation; returns list of decoded completions."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True) for p in prompts]
    out = []
    B = 16
    for i in range(0, len(texts), B):
        batch = texts[i:i + B]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(batch)):
            comp = gen[j][enc["input_ids"].shape[1]:]
            out.append(tok.decode(comp, skip_special_tokens=True))
    return out


def pick_qid(text, valid):
    """Extract the first Q/P id present in `valid` from the model output."""
    for m in re.findall(r"[QP]\d+", text):
        if m in valid:
            return m
    return None


def run_cta(args, kg, gen_fn, model_label):
    recs = list(load_dataset(args.dataset))
    gold = {r.table_id: r.gold_cta for r in recs}
    lab = lambda q: (kg.label(q) or q)  # noqa: E731
    prompts, keys, validsets, candlists = [], [], [], []
    for r in recs:
        for col in r.entity_columns:
            ents = [e for row in range(min(len(r.rows), MAXROWS)) if (e := r.gold_cea.get((row, col)))]
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
            prompts.append(p); keys.append((r.table_id, col))
            validsets.append(set(cands) | {cands[0]}); candlists.append(cands)
    print(f"[CTA] {len(prompts)} columns to classify", flush=True)
    outs = gen_fn(prompts)
    # build predictions (fallback to top-coverage candidate if parse fails)
    pred = {r.table_id: {} for r in recs}
    nfail = 0
    for (tid, col), text, valid, cands in zip(keys, outs, validsets, candlists):
        q = pick_qid(text, valid)
        if q is None:
            q = cands[0]  # FAIR fallback: top-coverage type (= counting's pick), not random
            nfail += 1
        pred[tid][col] = q
    print(f"  parse-failure fallback: {nfail}/{len(keys)} ({nfail/max(len(keys),1):.1%})", flush=True)
    for t in outs[:3]:
        print(f"  [sample-out] {t!r}", flush=True)
    m = evaluate_cta(pred, gold, kg)
    print("=" * 60)
    print(f"{args.dataset} CTA | LLM ({model_label}) constrained-choice | micro cscore")
    print(f"  LLM-CTA micro-F1 = {m['cta_micro_f1']:.3f}  macro = {m['cta_macro_f1']:.3f}")
    print("=" * 60)


def run_cpa(args, kg, gen_fn, model_label):
    recs = [r for r in load_dataset(args.dataset) if r.gold_cpa]
    gold = {r.table_id: r.gold_cpa for r in recs}
    plabels = json.loads((cache_dir() / "property_labels.json").read_text())
    plab = lambda p: plabels.get(p, p)  # noqa: E731
    prompts, keys, validsets = [], [], []
    for r in recs:
        nr = min(len(r.rows), MAXROWS)
        for i in r.entity_columns:
            for j in range(len(r.headers)):
                if j == i:
                    continue
                cand = Counter()
                for row in range(nr):
                    e = r.gold_cea.get((row, i))
                    if not e:
                        continue
                    o = r.gold_cea.get((row, j))
                    cell = r.rows[row][j] if j < len(r.rows[row]) else ""
                    for p, obj in kg.statements(e):
                        if o and obj == o:
                            cand[p] += 1
                    for p, val, kind in kg.literal_statements(e):
                        if literal_matches(cell, val, kind):
                            cand[p] += 1
                if not cand:
                    continue
                cands = [p for p, _ in cand.most_common(MAX_CANDS)]
                hi = r.headers[i] if i < len(r.headers) else ""
                hj = r.headers[j] if j < len(r.headers) else ""
                si = [r.rows[row][i] for row in range(min(6, len(r.rows))) if i < len(r.rows[row])]
                sj = [r.rows[row][j] for row in range(min(6, len(r.rows))) if j < len(r.rows[row])]
                opts = "\n".join(f"{p}: {plab(p)}" for p in cands)
                pr = (f"Column A (header \"{hi}\"): {', '.join(si)}\n"
                      f"Column B (header \"{hj}\"): {', '.join(sj)}\n\n"
                      f"Which ONE Wikidata property best describes the relation A->B? "
                      f"Choose from:\n{opts}\nor answer NONE if no relation holds.\n\nAnswer with only the P-id or NONE.")
                prompts.append(pr); keys.append((r.table_id, i, j)); validsets.append(set(cands))
    print(f"[CPA] {len(prompts)} column pairs to classify", flush=True)
    outs = gen_fn(prompts)
    pred = {r.table_id: {} for r in recs}
    for (tid, i, j), text, valid in zip(keys, outs, validsets):
        if "NONE" in text.upper()[:8]:
            continue
        q = pick_qid(text, valid)
        if q is not None:
            pred[tid][(i, j)] = q
    m = evaluate_cpa(pred, gold)
    print("=" * 60)
    print(f"{args.dataset} CPA | LLM ({model_label}) constrained-choice | exact micro-F1")
    print(f"  LLM-CPA micro-F1 = {m['cpa_micro_f1']:.3f}  (P={m['cpa_micro_p']:.3f} R={m['cpa_micro_r']:.3f})")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cta", "cpa"], required=True)
    ap.add_argument("--dataset", default="250wt")
    ap.add_argument("--model", default="qwen",
                    help="qwen (local 7B) | gpt-4o | gpt-4o-mini | claude-* (API, key from .env)")
    args = ap.parse_args()
    kg = CachedWikidataKG(offline=True)
    local = "qwen25-7b" if args.model == "qwen" else args.model
    if local in LOCAL_MODELS:
        tok, model = load_llm(local)
        gen_fn = lambda ps: chat_batch(tok, model, ps)  # noqa: E731
        label = LOCAL_MODELS[local][0]
    else:
        gen_fn = api_generator(args.model)
        label = args.model
    (run_cta if args.task == "cta" else run_cpa)(args, kg, gen_fn, label)


if __name__ == "__main__":
    main()
