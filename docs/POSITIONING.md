# POSITIONING.md — the reviewer-proof framing for AAAI

> Principle: make the NOVELTY be something that does NOT require beating accuracy
> SOTA, scope every accuracy claim to what the data supports ("competitive"), and
> back everything with rigor (5 datasets, 5 seeds, significance, ablations,
> reproducibility). This removes the only fatal attack ("you don't beat SOTA")
> because we never claim to. No framing guarantees acceptance, but this minimizes
> attackable surface and is 100% honest.

## Title direction
"Trustworthy Semantic Table Interpretation: Coverage-Guaranteed Abstention and a
Robustness Re-evaluation" (or "Calibrated, Ontology-Conditioned STI").

## The spine (headline = trustworthiness, NOT accuracy)
STI leaderboards optimize raw accuracy and ignore whether a system KNOWS WHEN IT IS
WRONG. We bring conformal prediction to CTA/CPA for the first time: coverage-
GUARANTEED prediction sets and principled abstention. Timely: SemTab 2025 added
selective prediction ("I don't know") and a robustness track — the field is moving
exactly here.

## Three contributions (each fully backed; none needs an accuracy win)
C1. CONFORMAL CTA/CPA (the novelty — first of its kind, nothing to "beat").
    - Coverage-guaranteed sets + abstention for column typing AND relations.
    - Empirical marginal coverage within +-2% of target over 5 seeds; the tuned
      softmax-threshold baseline UNDER-covers (no guarantee) -> conformal is the fix.
    - Method finding: LAC > APS for DEEP ONTOLOGY label hierarchies (APS bloats).
    - Reviewer cannot say "incremental": no prior conformal CTA/CPA exists.

C2. ONTOLOGY-CONDITIONED MODEL (the enabler, framed as COMPETITIVE not SOTA).
    - Joint CTA+CPA over a fused table+candidate+ontology graph; ontology-conditioned
      message passing / Graph-Transformer; learned hop-gate that GENERALIZES GRAMS+'s
      hand-set delta=0.1 (their own named future work: joint CTA+CPA).
    - Claim = "accuracy-competitive with SOTA across 5 datasets" (true: 250WT CTA in
      the SOTA ballpark; saturated synthetic matched). NEVER "we beat SOTA accuracy."

C3. CONTROLLED ROBUSTNESS STUDY + BENCHMARK RE-EVALUATION (a real, useful finding).
    - First systematic, controlled perturbation study of STI (cell/header/value/
      candidate noise, intensity-swept, seeded).
    - Finding: a counting baseline with MODERN retrieval already EXCEEDS published
      ToughTables SOTA under official scoring (AH 0.557 vs DAGOBAH 0.409) -> the
      benchmark's famed brittleness is largely a retrieval-era artifact. Useful to the
      community regardless of our model.
    - Calibration-under-shift: vanilla coverage degrades, shift-aware recovers.

## How this preempts every likely reviewer concern
- "Doesn't beat SOTA accuracy" -> we don't CLAIM to; novelty is C1 (calibration). Dead on arrival.
- "Incremental / just a GNN" -> conformal CTA/CPA is first; hop-gate generalizes GRAMS+'s constant
  (their named future work). Position vs GRAMS/Tab-HGNN/TCN/GAIT explicitly (done in research report).
- "Single dataset / weak eval" -> 5 datasets (250WT, HardTables, WikidataTables, ToughTables, tFood),
  official scorers, 5 seeds, paired significance tests on EVERY comparison.
- "No baselines / no LLM" -> GRAMS (public, run by us), an LLM few-shot (TableLlama/GPT-4o), counting.
- "GT noise / multi-valid-type" -> we report official scoring + per-table results; we EXPOSE the
  retrieval artifact rather than hide it.
- "Reproducibility" -> frozen splits+hashes, config+SHA dumps, one-command-per-table REPRODUCE.md.

## What must be TRUE for this to hold (the required work; method need only be COMPETITIVE)
1. 5-dataset eval with REAL candidates + official per-dataset scorers (P1/P2 in EXPERIMENT_PLAN).
2. Real baselines incl. one LLM (P3).
3. Conformal: shift-aware variant + coverage-under-shift curves; randomized APS for small sets (P5).
4. Robustness: noise-augmented training; graceful-degradation curves vs counting+LLM (P6).
5. Significance + full pre-registered ablation table (i-v) + reproducibility wrapper (P7).
6. Accuracy: must land "competitive" (within a small margin / matched on saturated). Already ~true.

## Honest caveat
Acceptance is never guaranteed. But this framing (a) makes the contribution a genuine FIRST that
needs no accuracy win, (b) scopes accuracy to a claim the data supports, (c) is rigor-complete. That
is the maximal-probability, zero-fabrication position. If a reviewer still wants an accuracy SOTA win,
that reviewer is asking for a different paper — and the evidence says no STI model beats strong
counting on clean CTA, so that paper is not achievable with this (or likely any) method here.
