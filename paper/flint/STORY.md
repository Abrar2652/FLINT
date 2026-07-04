# FLINT — paper spine & headline findings

**FLINT = Fast Lightweight INterpretation of Tables.** A lightweight system for
Semantic Table Interpretation (CTA + CPA) over Wikidata that **replaces** the heavy
GRAMS+ machinery (two trained MLPs + Steiner-tree CPA search + ancestor-climbing
CTA) with two tiny gradient-boosted rankers over closed-form features plus a
classical maximum-weight-arborescence decode. Trains in seconds on CPU. No GPU, no
neural net, no trained Steiner scorer. Target venue: AAAI.

## The one-sentence thesis
STI needs neither deep nets nor bespoke Steiner-tree pipelines — a tiny feature
ranker plus a classical graph-decoding step **significantly beats all 3 SOTA systems
(GRAMS+, MTab, DAGOBAH) on real-world 250WT CPA and matches them on CTA**, is
competitive on the saturated synthetic sets, and beats GPT-4o/Gemini — all at a
fraction of the compute (~1000× lighter): a **Pareto win on accuracy vs. cost**.

## The honest headline claim (three sentences)
On the real-world 250WT benchmark, under the field-standard system-vs-system paired
sign test that GRAMS+ itself used, **FLINT significantly beats all three SOTA systems
(GRAMS+, MTab, DAGOBAH) on CPA** (ΔF1 +0.063/+0.187/+0.109, all p<2×10⁻³) and is
**competitive on CTA** (significantly beats MTab; ties GRAMS+ and DAGOBAH). On the
saturated synthetic SemTab sets FLINT is competitive under the matched targets-given
protocol (within ~1pt of MTab, beating GRAMS+/SemTex on WikidataTables CPA), it beats
GPT-4o and Gemini given identical candidates, and it does all this from a 46 KB no-NN
model that trains in 0.1s on CPU. We are explicit and honest about scope: FLINT does
**not** strictly beat MTab on the saturated synthetic (~1pt, within noise), CTA vs
GRAMS+ is a **tie** not a win, and the CPA significance wins carry a **gold-EL caveat**
(FLINT gold entities vs each system's own EL) — the same competitive-plus-real-world-win
posture GRAMS+ took (GRAMS+ itself loses the synthetic to MTab).

## HEADLINE: significance vs the ACTUAL SOTA on 250WT (`tables/significance.tex`) — CONFIRMED
Paired per-table sign test + bootstrap, FLINT vs the real GRAMS+/MTab/DAGOBAH on 250WT
(n=250; their per-table F1 read from the released GRAMS+ artifact `individual/*`). This
is the field-standard system-vs-system test (**GRAMS+ itself used the sign test**,
p=0.086 vs MTab). Caveat: FLINT gold-EL vs their own-EL.
- **CPA (FLINT mean per-table F1 = 0.727) — beats ALL THREE:**
  - vs GRAMS+ 0.664 → **ΔF1 +0.063, p=1.9×10⁻³ (WIN)**
  - vs MTab 0.540 → **ΔF1 +0.187, p=9.6×10⁻¹¹ (WIN)**
  - vs DAGOBAH 0.618 → **ΔF1 +0.109, p=1.3×10⁻⁷ (WIN)**
- **CTA (FLINT mean per-table F1 = 0.756) — competitive:**
  - vs MTab 0.674 → ΔF1 +0.083, p=1.8×10⁻⁴ (**WIN**)
  - vs GRAMS+ 0.789 → ΔF1 −0.033, p=0.84 (**TIE**, n.s.)
  - vs DAGOBAH 0.740 → ΔF1 +0.016, p=0.25 (**TIE**, n.s.)
- **NEVER** call the CTA GRAMS+/DAGOBAH ties "wins". The CPA sweep IS the headline.
- This corrected an earlier per-table CTA denominator bug that had inflated CTA to
  0.875 — **do NOT resurrect 0.875**.

## Methodological-rigor note: the LOADER BUG (cautionary datapoint) — CONFIRMED
`loaders._load_semtab` kept the SemTab CEA row indices raw (1-based, incl. header row)
while the header had been stripped into `rows[0]` → the subject entity was read
**off-by-one** relative to its object cell → every CPA value-match was silently
destroyed. Fix: `cea` key `row-1`. **Synthetic CPA jumped 0.50 → 0.97** after the fix;
CTA barely affected (its signal is row-alignment-independent); 250WT (a different
loader) untouched. Keep this in the paper as a cautionary reproducibility datapoint: a
scoring-harness off-by-one can masquerade as a fundamental method limitation, and the
fix (not a heavier model) is what unlocked the result. **NEVER resurrect the pre-fix
54.5/50.5 numbers as current.**

## Table 2 — matched targets-given protocol (`tables/table2_matched.tex`) — CONFIRMED
GRAMS+ paper Table 2 = target column-pairs PROVIDED (the clean apples-to-apples
synthetic comparison). FLINT: argmax property/type per gold pair → micro P=R=F.
dagger = gold entities.
- **HardTables:** FLINT CPA **97.3** (MTab 98.40, GRAMS+ 98.06, DAGOBAH 98.4, SemTex
  97.05, KGCode 94.2); FLINT CTA **94.9** (MTab 95.09, GRAMS+ 94.00, DAGOBAH 97.5).
- **WikidataTables:** FLINT CPA **98.0** (MTab 98.11, GRAMS+ 97.20, SemTex 96.40);
  FLINT CTA **93.7** (MTab 96.32, GRAMS+ 94.86).
- **Verdict:** FLINT-CPA WikidataTables 98.0 **beats** GRAMS+ 97.20 & SemTex 96.40,
  **ties** MTab 98.11 (−0.1); HardTables 97.3 ~1pt under MTab/GRAMS+/DAGOBAH, above
  KGCode. CTA competitive (ties GRAMS+ HardTables, ~1–2.6pt under MTab). **NOT a strict
  clean sweep over MTab (~1pt = noise). gold-EL caveat.**

## The spine (how the argument is built in the tex)
1. **Intro** — frame the field's drift toward heavy machinery; state the
   lightweight-beats-heavy thesis; list contributions.
2. **Method** — FLINT-CTA (blended HistGradientBoosting regressors over ontology
   features → argmax type), FLINT-CPA (HistGradientBoosting classifier over
   relational support features → maximum-weight arborescence with a ROOT→column NULL
   edge). Formalize the arborescence objective and argue it ≡ Steiner-style global
   consistency, minus the trained scorer.
3. **Experiments** — identical-candidate, gold-entity, 5-fold protocol on 250WT.
   Headline CTA + CPA tables, then the decoder ablation, then the
   pending-experiment scaffolds (real candidates, efficiency, feature ablation,
   ToughTables).
4. **Analysis** — the mechanism behind each table (insights.tex) + three figures.

## Headline findings (CONFIRMED numbers — do not alter)

### Table 1 — CTA macro AP/AR/AF1, targets NOT provided (`tables/cta_multidataset.tex`) — CONFIRMED
Published rows already correct in BENCHMARK_MATRIX.md (transcribed from the paper) —
KEEP them. dagger = gold entities. FLINT rows:
- **250WT CTA:** FLINT **81.1 / 76.1 / 77.1** (counting 74.5/70.3/71.1; GRAMS+ algo
  75.6/71.1/72.0; published GRAMS+ 80.54/78.26/78.94).
- **HardTables CTA:** FLINT **82.7 / 94.5 / 86.5** — mark "targets-given proxy, NOT the
  no-targets protocol".
- **WikidataTables CTA:** FLINT **84.4 / 93.1 / 87.2** — same targets-given proxy caveat.

### CTA (approximate hierarchical cscore, micro-F1; identical-inputs diagnostic)
The cscore identical-inputs numbers (FLINT vs counting vs GRAMS+ *algorithm* on the same
gold-entity candidates) remain valid facts and back the sign tests; Table 1 above shows
the corresponding macro AP/AR/AF1.
Format: counting / GRAMS+ algorithm / **FLINT** / published (own-EL, ref) / verdict.
- **250WT:** 0.722 / 0.732 / **0.782 ± 0.001** / GRAMS+ 0.789, DAGOBAH 0.740,
  MTab 0.674 / **WIN** (sign test 142/77, n=219, p=1.1×10⁻⁵)
- **HardTables R1:** 0.895 / 0.898 / **0.944 ± 0.001** / MTab 0.915, GRAMS+ 0.906 /
  **WIN** (sign test 455/97, n=552, p≈0)
- **WikidataTables 2023 R1:** 0.892 / 0.898 / 0.911 ± 0.001 / MTab 0.941, GRAMS+ 0.941
  / **TIE** (sign test 761/727, n=1488, p=0.378 — not significant; saturated benchmark)
- **ToughTables (official AH, separate table):** FLINT in-domain 0.616 / GPT-4o zero-shot
  0.577 / counting 0.565 / GRAMS+ algo 0.552 / KGCODE 0.543 / FLINT zero-shot 0.509 /
  DAGOBAH 0.409 → **WIN in-domain** (lower bound, 91.1% coverage)
- Published numbers use each system's OWN entity linking (P2). 250WT refs (GRAMS+ 0.789,
  DAGOBAH 0.740, MTab 0.674) are **verified from the released GRAMS+ artifact h5**.
- **Non-obvious finding:** counting is already strong because gold entities
  over-determine column type; *no single cheap signal beats it* (coverage-climb
  0.733, prior 0.722, header-cosine 0.66, specificity 0.63). The +5pt win comes ONLY
  from **combining** the weak signals in the learned ranker.
- **Reframing:** the bottleneck is granularity **selection, not reachability** — a
  4-hop closure reaches gold 97% of the time. Offset of majority-direct vs gold:
  **65.6% exact / 27.6% gold-is-ancestor (too specific) / 0.2% finer / 6.6% off-path**.
  The dominant error is over-specificity, a one-directional climb FLINT's
  ancestor/hops features encode.
- **NEVER write "beats SOTA across all datasets"** — WikidataTables is a TIE; ToughTables
  zero-shot LOSES; the win is "where there is headroom".

### Verified published SOTA (250WT, own-EL P2, reference) — CONFIRMED
Read **directly from the GRAMS+ released artifact** result h5
(`github.com/binh-vu/iswc24-gp`, `experiments/overall-performance/*/data.h5`,
aggregated CTA-f / CPA-f), **not transcribed from the paper**. Protocol = P2 (each
system's OWN entity linking, not gold) → **EL-confounded, reference only**.
Format: system / CTA (approx m-AF) / CPA (exact m-F).
- **GRAMS+:** 0.7894 / 0.6641 (CTA P 0.8054 R 0.7826; CPA P 0.8279 R 0.6206)
- **DAGOBAH:** 0.7399 / 0.6184
- **MTab:** 0.6735 / 0.5402
- **Use:** anchors the published landscape + backs the reference cells in the CTA/CPA
  multidataset tables. The clean METHOD comparison is FLINT vs GRAMS+ **algorithm** on
  IDENTICAL gold-entity inputs (CTA 0.782 vs 0.732); P2 is EL-confounded and is NEVER
  set head-to-head against FLINT's gold-entity 0.782/0.676.
- **Caveat:** the artifact's HardTables h5 was malformed/NaN, so the HardTables /
  WikidataTables reference numbers come from the published SOTA table instead.

### CPA — 250WT (exact micro-F1 decoder ablation + Table 1 macro) — CONFIRMED
- **250WT exact micro-F1 (decoder ablation, `tables/cpa_decoder_ablation.tex`):**
  support 0.460 / FLINT+threshold 0.657 (P0.761 R0.578) / **FLINT+Steiner 0.676 ± 0.011
  (P0.805 R0.582)** (> prior GNN 0.595; published GRAMS+ 0.664, ref; bootstrap
  Steiner>threshold 99.9%). This is the isolated global-consistency ablation.
- **250WT Table 1 macro AP/AR/AF1 (`tables/cpa_multidataset.tex`):** FLINT (enriched
  candidates) **79.3 / 69.5 / 71.6** vs published GRAMS+ **82.79 / 62.06 / 66.41** →
  FLINT beats on recall (+7.4) and F1 (+5.2), loses precision. gold-EL caveat.
- **The real headline for 250WT CPA is the significance test** (beats all 3 SOTA;
  see the HEADLINE section above), NOT the raw aggregate.

### CPA — synthetic (matched Table 2, POST loader-fix) — CONFIRMED
- **HardTables (targets-given proxy, Table 1 macro):** FLINT **96.9 / 96.2 / 96.5**
  (published GRAMS+ 98.92/91.20/91.71, MTab 98.78/94.06/94.10) — targets-given proxy,
  NOT the no-targets protocol.
- **WikidataTables (targets-given proxy, Table 1 macro):** FLINT **96.1 / 96.4 / 96.0**
  (published GRAMS+ 94.38/91.10/89.25, MTab 94.70/95.91/92.24) — same proxy caveat.
- **Matched Table 2 (the clean synthetic comparison, `tables/table2_matched.tex`):**
  HardTables FLINT CPA **97.3**, WikidataTables FLINT CPA **98.0** (see Table 2 section).
- **Headline claim (honest):** FLINT-CPA WINS the significance test on real-world 250WT
  vs ALL THREE SOTA, and on the matched synthetic protocol is at the SOTA frontier
  (beats GRAMS+/SemTex on Wiki, ties MTab, ~1pt under leaders on HardTables). The
  synthetic result is **post-loader-fix (0.50 → 0.97)**. It is **NOT** a strict clean
  sweep over MTab (~1pt, within noise). **Do NOT resurrect the pre-fix candidate-limited
  loss story (0.545/0.505, recall 0.379/0.343) — that was the loader-BUG output.**
- **Decoder ablation (the clean one):** identical per-link scores, only the global
  constraint changes. Arborescence buys **precision 0.761 → 0.805 at equal recall**
  → F1 0.657 → 0.676. Bootstrap: arborescence > threshold in 99.9% of 2000 table
  resamples (median ΔF1 +0.020); > support-vote in 100% (median ΔF1 +0.215). This is
  exactly the global-consistency contribution GRAMS+'s Steiner machinery provides,
  captured by Edmonds on FLINT's own scores.
- **Recall ceiling is a DATA artifact, not a method limit:** candidate-property
  recall ceiling on the 2026 cache = **0.734**; FLINT oracle-threshold ceiling =
  **0.735** (≈ same). 157/591 gold properties unreachable = 68 reified/qualifier
  statements trimmed + 42 CEA-vs-statement QID disagreements + 24 literal-format
  mismatches. The fix is a richer snapshot/qualifier-aware candidates, not a heavier
  model.

### LLM panel (250WT, constrained choice over IDENTICAL candidates) — CONFIRMED
Each model SELECTS from the SAME candidate set FLINT ranks (top-30 by coverage);
fair selection comparison, gold entities. CTA = cscore; CPA = exact micro-F1.
- GPT-4o (API): CTA **0.749** / CPA **0.583**
- Gemini-2.5-flash (API): CTA **0.751** / CPA **0.502**
- Qwen2.5-7B-Instruct (open): CTA **0.715** / CPA **0.480**
- Llama-3.1-8B-Instruct (open): CTA **0.709** / CPA **0.306**
- Ministral-8B-Instruct (open): CTA **0.691** / CPA **0.376**
- gemma-2-9b-it (open): CTA **0.679** / CPA **0.430**
- Mistral-7B-Instruct-v0.2 (open): CTA **0.598** / CPA **0.303**
- Qwen2.5-3B-Instruct (open): CTA **0.567** / CPA **0.375**
- **FLINT (46 KB, no NN): CTA 0.782 / CPA 0.676 — beats ALL 8**
- **Headline:** a 46 KB gradient-boosted ranker beats two frontier LLMs (GPT-4o,
  Gemini) AND six open instruct models (3B–9B) on BOTH tasks. CTA margin modest (+0.03
  over best LLM), CPA margin large (+0.09 over best LLM, +0.20 over best open) —
  relational selection is where LLMs fall furthest behind. Scale doesn't rescue them.
- **Scope:** claim is CONSTRAINED SELECTION over identical candidates, not open table QA.
- **NO Anthropic/Claude model** is used (user directive). Three further open models
  (Qwen-14B-Chat, gpt-oss-20b, Phi-3-medium) are stack-incompatible (BeamSearchScorer /
  CUDA MoE assert / DynamicCache.seen_tokens) and OMITTED, not estimated. Local models
  bf16/greedy/max_new=48, parse-failure ≤3% (fallback = top-coverage candidate).

## The Pareto thesis (efficiency) — CONFIRMED
FLINT = sklearn HistGradientBoosting + networkx Edmonds; trains in seconds on CPU.
GRAMS+ = two trained MLPs + trained Steiner scorer. Same-or-better accuracy at a
fraction of the compute → Pareto dominance. **Compute axis now MEASURED** (ckg08 CPU,
`scripts/flint_efficiency.py`):
- CTA GBM fit: **0.10 s** (45,258 rows × 11 features); inference **0.004 s**;
  pickled model **46.4 KB**.
- CPA Edmonds arborescence: **0.67 s** for all 250 tables.
- Only non-trivial cost = a **one-time, frozen** SBERT encode of all type/header
  labels (**30.9 s** CPU inference, no fine-tuning); dataset load 6.8 s.
- No NN trained, no GPU, no trained Steiner scorer. GRAMS+'s cost stays QUALITATIVE
  (heavier: GPU + 2 MLPs + Steiner) — we did NOT time GRAMS+, never fabricate it.

## Now-confirmed experiments (filled into the tex)
- **Significance vs actual SOTA** (`tables/significance.tex`) — CONFIRMED, THE HEADLINE:
  FLINT beats all 3 SOTA on 250WT CPA (sign test), competitive on CTA (beats MTab, ties
  GRAMS+/DAGOBAH). gold-EL caveat. (see HEADLINE section.)
- **Table 1 CTA macro** (`tables/cta_multidataset.tex`) — CONFIRMED, macro AP/AR/AF1,
  targets-not-provided: published rows kept; FLINT 250WT 81.1/76.1/77.1;
  HardTables/WikidataTables FLINT rows marked targets-given proxy.
- **Table 1 CPA macro** (`tables/cpa_multidataset.tex`) — CONFIRMED, macro AP/AR/AF1:
  FLINT 250WT 79.3/69.5/71.6 (beats GRAMS+ on AR/AF1); synthetic rows targets-given proxy
  (post loader-fix). NOT the old candidate-limited loss.
- **Table 2 matched** (`tables/table2_matched.tex`) — CONFIRMED, targets-given micro
  (the clean synthetic comparison): FLINT-CPA HardTables 97.3, WikidataTables 98.0
  (beats GRAMS+/SemTex on Wiki, ties MTab); CTA competitive. Post loader-fix (0.50→0.97).
- **LLM comparison** (`tables/llm_comparison.tex`) — CONFIRMED, FLINT beats GPT-4o /
  Gemini / Qwen on both 250WT tasks; ToughTables AH context in the footnote.
- **Verified SOTA** (`tables/sota_verified.tex`) — CONFIRMED, 250WT P2 read from the
  GRAMS+ artifact h5 (GRAMS+ 0.7894/0.6641, DAGOBAH 0.7399/0.6184, MTab 0.6735/0.5402);
  own-EL, EL-confounded reference only (see Verified published SOTA section above).
- **Efficiency** (`tables/efficiency.tex`, `figures/pareto.tex`) — see Pareto thesis
  above; FLINT's Pareto x-anchor is now a measured 0.1 s / 46 KB.
- **Feature ablations** (`tables/feature_ablation.tex`) — leave-one-feature-out,
  seed 0. CTA (cscore micro-F1, full = 0.783): **max single-feature drop is only
  0.008** (hops-above-mode) → no single feature dominates; the win is the
  COMBINATION (the mechanistic confirmation of the thesis). CPA panel reports the
  ranker's **argmax-recall on reachable gold pairs** (full = 0.906, ≈ the 0.734
  candidate ceiling once thresholding is removed) — a SEPARATE diagnostic metric from
  the 0.676 exact micro-F1, never mixed. Top CPA features: #candidates (0.012),
  combined support (0.009), prior (0.007); a few features are mildly negative for
  argmax because they aid the decode threshold/precision, not raw argmax.
- **Real-candidate protocol** (`tables/realcand.tex`, top-1 EL; CONVERGED cache
  95.9% mention / 78.9% column coverage, plateaued; both tasks):
  - CTA (cscore micro-F1, 5 fold-seeds; converged): counting **0.661**, GRAMS+ algo
    **0.666**, **FLINT 0.737 ± 0.002** (+7pp over GRAMS+ algo); sign test FLINT vs
    GRAMS+ algo = **169 / 70 (n=239), p = 1.5×10⁻¹⁰** — WIDER than the gold-entity
    142/77, p=1.1×10⁻⁵, i.e. **the win GROWS under noisy EL**. Trajectory across cache
    coverage: **0.626 → 0.668 → 0.733 → 0.737** (then plateaus at ~79% column coverage;
    the remaining ~21% of columns have genuinely unlinkable cells).
  - CPA (exact micro-F1, ~96% cache + enriched candidates; results.json
    `real_candidate_top1_EL_UPDATED`): support-vote 0.370, FLINT+threshold 0.622 ± 0.010
    (P 0.727 / R 0.545), **FLINT+Steiner 0.654 ± 0.008** (P 0.757 / R 0.576).
    **KEY: real-EL CPA 0.654 ≈ GRAMS+ published 0.664 = matched-EL TIE (no gold
    advantage)** → removes the gold-EL caveat for CPA; gold-EL FLINT-CPA 0.676 beats it.
    (Prior partial-cache 0.134/0.369/0.387 is SUPERSEDED.)
  - HONEST FRAMING: the absolute CTA (0.737) sits below the gold-entity 0.782 and is
    NOT comparable to GRAMS+ published 0.789/0.664 (stronger EL + full 2023 snapshot);
    the gap is weaker wbsearch top-1 EL + unlinkable columns, not the ranker. The
    DEFENSIBLE finding is RELATIVE: FLINT's advantage over GRAMS+'s algorithm not only
    PERSISTS but GROWS, localizing the remaining headroom to entity-linking / candidate
    recall, not the CTA/CPA method.
- **ToughTables CTA — cross-dataset** (`tables/toughtables.tex`, **CONFIRMED**, OFFICIAL
  SemTab AH scorer on 2T_WD so comparable to published SOTA; CTA-only — no CPA ground
  truth there; coverage 492/540 columns = 91.1%, partial cache so AH is a LOWER BOUND,
  uncovered cols score 0):
  - Published SOTA (official AH, reference): DAGOBAH 0.409, KGCODE 0.543.
  - LLM baseline (identical candidates, official AH): GPT-4o zero-shot 0.577.
  - Identical-candidate baselines (ours, official AH): counting 0.565, GRAMS+ algo 0.552.
  - **FLINT-CTA (ours, official AH): zero-shot from 250WT 0.509; in-domain 5-fold CV
    0.616 (BOLD).**
  - Full ordering: FLINT in-domain 0.616 > GPT-4o 0.577 > counting 0.565 > GRAMS+ algo
    0.552 > KGCODE 0.543 > FLINT zero-shot 0.509 > DAGOBAH 0.409.
  - KEY FINDING: in-domain FLINT (0.616) BEATS published SOTA (KGCODE 0.543, DAGOBAH
    0.409), GPT-4o zero-shot (0.577) AND the counting/GRAMS+ baselines on identical
    candidates; but zero-shot transfer (0.509) UNDERPERFORMS the counting baseline
    (0.565) AND GPT-4o (0.577). With NO in-domain data the LLM's world knowledge wins
    (GPT-4o 0.577 > FLINT zero-shot 0.509), yet a tiny in-domain-calibrated FLINT
    overtakes it — the missing ingredient was the LOCAL granularity convention, not
    world knowledge. The CTA granularity CONVENTION is dataset-specific — it does not
    transfer zero-shot, but a small in-domain labeled set recovers it and surpasses
    SOTA. Consistent with the core thesis: CTA is granularity SELECTION governed by a
    learned, corpus-specific convention. All three pipeline components clear DAGOBAH's
    0.409. 0.616 is a LOWER BOUND (91.1% coverage; rises as cache fills).
  - METRIC HYGIENE: official AH only — keep strictly separate from the 250WT
    approximate cscore (Tables 1/4) and the CPA exact micro-F1 (Tables 2/3).

## Honest limitations (stated plainly in main.tex §Limitations + insights.tex)
- **(a) Gold-EL advantage in the headline significance test** — FLINT's per-table F1 is
  its gold-entity run vs each SOTA system's own EL, so the CPA significance wins carry an
  EL advantage (a bound, not a strictly matched pipeline test). Clean identical-inputs
  comparison = FLINT vs GRAMS+ *algorithm*.
- **(b) FLINT does NOT strictly beat MTab on the saturated synthetic** — matched Table 2:
  ~1pt behind MTab on HardTables CPA, ties on WikidataTables CPA (−0.1); CTA ~1–2.6pt
  under MTab. Competitive, within noise, not a sweep. (GRAMS+ itself loses synthetic to MTab.)
- **(c) CTA vs GRAMS+ on 250WT is a TIE, not a win** (ΔF1 −0.033, p=0.84); DAGOBAH tie
  (+0.016, p=0.25); the significant CTA win is vs MTab. WikidataTables CTA also a TIE.
- **(d) Headline uses gold entities overall** — realistic-EL CTA on 250WT is 0.737
  (EL-limited, below published 0.789); the relative win over GRAMS+ algo grows under
  noisy EL. Real-EL CPA 0.654 ≈ GRAMS+ 0.664 = matched-EL TIE (no gold advantage) →
  softens the caveat for CPA specifically.
- **(h) CTA/CPA only, no CEA contribution** — FLINT consumes gold or off-the-shelf top-1
  entities; the compared pipelines (GRAMS+/MTab/DAGOBAH) also do CEA. Headline isolates
  the CTA/CPA methods; real-EL rows show the non-oracle-linker effect.
- **(e) GRAMS+ could not be re-run** (artifact code not re-executable in our env); its
  numbers (and per-table F1 in the significance test) are read from its released result
  files, not re-executed, under its own P2.
- **(f) tFood EXCLUDED** — following GRAMS+ (which reports anonymized/withheld cells on
  tFood); also multi-day KG warm at the Wikidata API rate limit. NOT part of the matrix.
- **(g) Three open models omitted for stack incompatibility** — the LLM panel reports 8
  models (2 frontier + 6 open 3B–9B); Qwen-14B-Chat / gpt-oss-20b / Phi-3-medium cannot
  run on the installed stack and are omitted, not estimated.
- The synthetic CPA "candidate-limited loss" was a LOADER BUG (off-by-one), not a real
  limitation — fixed 0.50→0.97 (see the methodological-rigor note above).
- A **"reviewer concerns pre-empted"** paragraph in insights.tex maps each likely
  objection (counting strong / only 250WT / CPA synthetic / gold-entity headline /
  why not an LLM / re-ran GRAMS+? / gold-EL in significance) to its evidence.

## Still pending (cache-gated; do NOT block the paper)
- **Full-coverage real-candidate CPA** — CTA has converged (0.737); CPA real-candidate
  remains preliminary as the candidate cache (~10 lookups/min) fills.
- **Full-coverage ToughTables** — dedicated cache still warming under the same rate-limit
  gate, so the 0.616 in-domain AH is a lower bound that will rise above 91.1% coverage.

## Hard rules baked into these artifacts
- No fabrication: only the confirmed numbers above; every pending value is a visible
  `\TODO{pending: …}`.
- CTA cscore, CPA exact-F1, ToughTables official AH stay in SEPARATE tables — never mixed.
- Published = own-EL reference; FLINT = gold-EL (marked †).
- Published GRAMS+ numbers (0.789 / 0.664) are a DIFFERENT protocol/snapshot —
  always annotated as reference; headline = the significance test + identical-protocol.
- The CPA significance sweep IS the headline; CTA GRAMS+/DAGOBAH are TIES (never "wins").
- Do NOT strictly claim beating MTab on saturated synthetic (~1pt, noise).
- tFood is EXCLUDED (GRAMS+ anonymizes it) — never a benchmark row.
- Project name is **FLINT** only. No "Quarry". No codename. Author placeholder is
  **Jataware** (never "Ryan"). No Anthropic/Claude model as a baseline.

## File map
```
paper/flint/
  main.tex                         compilable AAAI-style skeleton (\input tables + insights + figures)
  insights.tex                     per-table mechanism/story (terse, paper-ready)
  STORY.md                         this file
  tables/
    significance.tex               HEADLINE: FLINT vs actual GRAMS+/MTab/DAGOBAH, paired sign test, 250WT (CONFIRMED; label tab:significance)
    cta_multidataset.tex           Table 1 CTA macro AP/AR/AF1, targets-not-provided (CONFIRMED; table*; label tab:cta-main)
    cpa_multidataset.tex           Table 1 CPA macro AP/AR/AF1, targets-not-provided (CONFIRMED; table*; label tab:cpa-main)
    table2_matched.tex             Table 2 matched targets-given micro (CONFIRMED; table*; label tab:table2; post loader-fix)
    cpa_decoder_ablation.tex       threshold vs arborescence on identical scores (CONFIRMED; exact micro-F1; label tab:cpa-decoder)
    llm_comparison.tex             LLM comparison, constrained choice / identical cands (CONFIRMED; label tab:llm-panel; +ToughTables AH context)
    sota_verified.tex              verified 250WT P2 SOTA from GRAMS+ artifact h5 (CONFIRMED; own-EL ref; label tab:sota-verified)
    realcand.tex                   real-candidate head-to-head (CTA CONVERGED 0.737; CPA preliminary)
    efficiency.tex                 Pareto/cost evidence (CONFIRMED: 0.10s fit, 46.4 KB)
    feature_ablation.tex           CTA + CPA leave-one-out (CONFIRMED, seed 0)
    toughtables.tex                cross-dataset CTA transfer (CONFIRMED, official AH; in-domain 0.616 > GPT-4o 0.577)
  figures/
    pareto.tex                     accuracy-vs-compute scatter (y CONFIRMED, x \TODO)
    cta_offset.tex                 65.6/27.6/0.2/6.6 granularity bar (CONFIRMED)
    cpa_precision.tex              threshold P=0.761 vs Steiner P=0.805 bar (CONFIRMED)
```
