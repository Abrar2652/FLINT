# POSITIONING_LITREVIEW.md — novelty & related-work audit (deep-research, 2026-06-11)

> De-risking pass for the AAAI submission. Citations below appeared in live web
> searches (real, not fabricated) BUT exact authors/year/venue must be re-verified
> at cite-time via `/ars-citation-check` before they enter the paper (gray zone = FAIL).

## Novelty verdict per contribution

### C1 — Conformal coverage-guaranteed CTA/CPA + abstention  → **NOVEL (with required differentiation)**
No prior work applies formal conformal prediction (coverage guarantee) to CTA or CPA. The
intersection "conformal prediction + semantic table interpretation" has no substantial publications.
BUT three neighbors must be cited and differentiated, or a reviewer calls it derivative:
- **Bono/Belotti et al., "Efficient Uncertainty Estimation for LLM-based Entity Linking in Tabular
  Data"** (arXiv:2510.01251, OM@ISWC 2025). CLOSEST. Differentiator: theirs is heuristic uncertainty
  *estimation* for **CEA/entity-linking**, NOT a formal coverage guarantee and NOT CTA/CPA. Ours is
  finite-sample conformal coverage on CTA/CPA.
- **SemTab 2025 selective-prediction ("I don't know") + Secu-Table robustness track** (CEUR
  Vol-4144). Differentiator: their selective mode = LLM *confidence-threshold* abstention (no
  guarantee); ours = conformal sets with a coverage guarantee. This is a TAILWIND (field is moving
  here) — cite as motivation, differentiate on the guarantee.
- **CF-GNN** (Huang et al., NeurIPS 2023, arXiv:2305.14535). Methodological backbone; differentiator:
  generic transductive node classification, not CTA/CPA, not a per-column variable label space.

### C1-sub — "LAC > APS for deep ontology hierarchies"  → **NOT NOVEL as a conformal claim → DOWNGRADE**
The LAC (small sets, weaker conditional coverage) vs APS (adaptive, larger sets) trade-off and
hierarchical conformal are established:
- "Conformal Prediction in Hierarchical Classification" (arXiv:2501.19038)
- "Softmax is not Enough (for Adaptive Conformal Classification)" (arXiv:2602.19498)
RISK: do NOT claim this as a novel conformal insight. Present as an *empirical observation in the STI
setting* (deep ontology-ancestor label space inflates non-randomized APS), and cite the hierarchical-
conformal lit. Randomized APS as the fix is standard (Romano 2020) — cite, don't claim.

### C2 — Ontology-conditioned joint CTA+CPA GNN (competitive accuracy)  → **INCREMENTAL (as expected)**
Joint/relational table models exist (GRAMS/GRAMS+, Tab-HGNN, TCN, GAIT, Doduo, TURL, KGLink). The
hop-gate generalizing GRAMS+'s delta=0.1 is a genuine but small delta. Position as "competitive
enabler," not a contribution headline (already our locked stance). Must differentiate from GRAMS+
(their named future work = joint CTA+CPA) explicitly.

### C3 — ToughTables brittleness is largely a retrieval-era artifact  → **NOVEL (likely) — verify framing**
No paper makes the exact claim "a counting baseline with modern retrieval exceeds published ToughTables
SOTA under official AH." Closest neighbors to differentiate:
- **Belotti et al., "Evaluating LLMs on Entity Disambiguation in Tables"** (arXiv:2408.06423) — re-eval
  of EL on hard tables with LLMs; differentiator: they study CEA/EL with LLMs, not a counting-baseline
  CTA re-eval / retrieval-artifact thesis.
- **"Robust LLM-based CTA via Prompt Augmentation with LoRA"** (arXiv:2512.22742) — recent robust-CTA;
  overlaps our robustness angle, MUST cite + differentiate (theirs = LLM+LoRA prompt aug; ours =
  controlled perturbation study + calibrated abstention + the retrieval-artifact diagnosis).
- "Evaluating Knowledge Generation/Self-Refinement for LLM-based CTA" (arXiv:2503.02718) — LLM-CTA, cite.

## Must-cite / must-differentiate set (8)
1. CF-GNN — Huang et al. NeurIPS 2023 (2305.14535) — conformal-GNN backbone.
2. Bono/Belotti — LLM-EL uncertainty in tables (2510.01251) — closest uncertainty-in-STI.
3. SemTab 2025 results (CEUR Vol-4144) — selective prediction + robustness track (timeliness).
4. GRAMS+ — Vu et al. ISWC 2024 — our base; their future work = joint CTA+CPA.
5. Korini & Bizer — CTA-with-ChatGPT (2306.00745) + CPA-with-LLMs (Springer 2024) — LLM-STI baselines.
6. ToughTables — Cutrona et al. (Zenodo 7419275) — the benchmark we re-evaluate.
7. Hierarchical conformal (2501.19038) + Romano APS — for the LAC/APS framing (so we don't overclaim).
8. Robust LLM-CTA + LoRA (2512.22742) — closest robustness competitor.

## Novelty risks to preempt (in the paper)
- R1: "LAC>APS" overclaim → reframe as STI-specific empirical observation + cite hier-conformal. (HIGH)
- R2: Reviewer says "SemTab 2025 already does selective prediction" → differentiate guarantee vs LLM
  confidence; we are the FORMAL coverage version + CTA/CPA (they were CEA/LLM-confidence). (HIGH)
- R3: "uncertainty in tables already exists (Bono)" → CEA-estimation vs CTA/CPA-guarantee. (MED)
- R4: C3 retrieval-artifact must use OFFICIAL scoring + report candidate-recall + acknowledge it is a
  baseline (not our model) win, else "unfair comparison". (MED — already handled in our table.)
- R5: LLM baseline is now MANDATORY (Korini&Bizer/TableLlama) — reviewers expect it. (HIGH — build it.)

## Bottom line
Headline (C1 conformal CTA/CPA) survives as a genuine FIRST, *if* differentiated from Bono/SemTab-2025/
CF-GNN and the LAC/APS sub-claim is downgraded. C3 (retrieval-artifact) is a fresh, useful finding.
C2 is a competitive enabler. The framing is defensible; the gating to-dos are the LLM baseline (R5)
and the LAC/APS reframe (R1).
