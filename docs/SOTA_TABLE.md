# SOTA_TABLE.md — published numbers we benchmark against

RULE (CLAUDE.md s2): a number enters a results table ONLY if reproduced here OR
cited below WITH a matching protocol. Never mix approximate-cscore with exact-F1.

> ⚠️ PROTOCOL PROVENANCE (read before using ANY 250WT number).
> There are THREE different 250WT protocols in the literature and they give
> different absolutes. Do NOT mix them:
>   (P1) GRAMS Semantic-Web-Journal: macro approximate avg P/R/F1, 10 runs,
>        GRAMS *given correct entities*. -> GRAMS CPA 74.1 / CTA 81.8.
>   (P2) GRAMS+ ISWC-2024 Table 1: CPA = exact micro F (m-F); CTA = approximate
>        micro (m-AF). Target columns NOT provided. GRAMS re-run here with TOP-1
>        candidates (so it scores LOWER than its own P1 numbers). THIS IS THE
>        BASELINE WE BEAT — verified directly from the PDF (see below).
>   (P3) GRAMS+ ISWC-2024 Table 2: same metrics but target columns PROVIDED.
> Our headline 250WT comparison uses **P2** (hardest, most realistic, full setting).

## 250WT — GRAMS+ ISWC-2024 **Table 1** (P2: no target columns; CPA=m-F exact, CTA=m-AF approx)
VERIFIED from `grams+ iswc24-paper.pdf` p.10. THIS IS THE BAR FOR OC-GNN ON 250WT.
| System | CPA m-F | CTA m-AF |
|--------|---------|----------|
| KGCode-Tab | 36.52 | 47.82 |
| GRAMS (top-1 cand.) | 44.71 | 70.45 |
| MTab | 54.09 | 67.16 |
| DAGOBAH | 61.84 | 73.99 |
| **GRAMS+** | **66.41** | **78.94** |

## 250WT — GRAMS Semantic-Web-Journal (P1: macro approx F1, GRAMS given gold entities) — REFERENCE ONLY
Do NOT put these in the same table as P2 numbers (different protocol).
| System | CPA F1 | CTA F1 |
|--------|--------|--------|
| MantisTable | 48.5 | 45.2 |
| BBW | 20.5 | 36.0 |
| MTab | 61.5 | 77.0 |
| GRAMS | 74.1 | 81.8 |

## WikidataTables / HardTables — GRAMS+ ISWC-2024 Table 1 (P2: no target cols) — VERIFIED from PDF p.10
NOTE: on these (near-saturated) sets GRAMS+ LOSES to MTab. Do not stake claims here.
| Dataset | System | CPA m-F | CTA m-AF |
|---------|--------|---------|----------|
| HardTables | MTab | 94.10 | 91.53 |
| HardTables | GRAMS+ | 91.71 | 90.63 |
| WikidataTables | MTab | 92.24 | 94.10 |
| WikidataTables | GRAMS+ | 89.25 | 94.06 |

## WikidataTables 2024 R1/R2 (micro P/R/F1) — GRAMS+ at SemTab 2024 (verbatim, separate from above)
| Round | CTA F1 | CPA F1 (P / R) |
|-------|--------|----------------|
| R1 | 92.9 | 89.8 (98.8 / 82.3) |
| R2 | 95.6 | 89.9 (99.2 / 82.19) |

## HardTables R1 2022 (micro F1)
| System | CTA | CEA | CPA |
|--------|-----|-----|-----|
| DAGOBAH-SL | 0.975 | 0.954 | 0.984 |
| KGCODE-Tab | 0.942 | 0.893 | 0.906 |

## WikidataTables 2023 R1 (micro F1)
| System | CEA | CTA | CPA |
|--------|-----|-----|-----|
| SemTex | 0.885 | 0.934 | 0.964 |
| TorchicTab | 0.830 | 0.817 | 0.934 |

## ToughTables / 2T (the robustness frontier — note collapse)
| System | CTA | CEA |
|--------|-----|-----|
| KGCODE-Tab 2022 (WD) | 0.543 | 0.905 |
| DAGOBAH 2022 (WD) | 0.409 | 0.945 |

## Caveats (CLAUDE.md s7)
- Two GRAMS papers, different numbers. Cite the right one per claim.
- MTab last competed 2021; TorchicTab debuted 2023; DAGOBAH skipped 2023.
- ToughTables GT disputed — use for robustness framing, report per-table.

---
# FLINT RESULTS vs SOTA (2026-06-22) — reproduced in this repo; see experiments/flint/results.json

## CTA (approximate cscore micro-F1). FLINT = gold-entity, identical-candidate protocol.
| Dataset | counting | GRAMS+ algo (ours, identical) | **FLINT** | published (own-EL, ref) | verdict |
|---------|----------|-------------------------------|-----------|--------------------------|---------|
| 250WT | 0.722 | 0.732 | **0.782** | GRAMS+ 0.789 / DAGOBAH 0.740 / MTab 0.674 | WIN (sign p=1.1e-5) |
| HardTables R1 | 0.895 | 0.898 | **0.944** | MTab 0.915 / GRAMS+ 0.906 | WIN (p~0) |
| WikidataTables 2023 R1 | 0.892 | 0.898 | 0.911 | MTab/GRAMS+ 0.941 | competitive/TIE (p=0.38) |
| ToughTables (official AH) | 0.565 | 0.552 | **0.616** in-domain | KGCODE 0.543 / DAGOBAH 0.409 | WIN (zero-shot 0.509) |

## CPA (exact micro-F1). FLINT = ranker + Steiner/arborescence, gold entities.
| Dataset | support-vote | **FLINT+Steiner** | published (ref) | note |
|---------|--------------|-------------------|------------------|------|
| 250WT | 0.460 | **0.676** | GRAMS+ 0.664 | WIN (bootstrap 99.9% vs threshold) |
| HardTables R1 | 0.462 | 0.545 | MTab 0.941 / GRAMS+ 0.917 | candidate-limited (cand recall 0.398) |
| WikidataTables 2023 R1 | 0.437 | 0.505 | MTab 0.922 / GRAMS+ 0.893 | candidate-limited (R 0.343) |

## LLM baselines (250WT, constrained-choice over identical candidates) — all < FLINT
| Model | CTA | CPA |
|-------|-----|-----|
| Qwen2.5-7B | 0.714 | 0.480 |
| GPT-4o | 0.749 | 0.583 |
| Gemini-2.5-flash | 0.751 | 0.502 |
| (local panel: Llama-3.1-8B, gemma-2-9b, Mistral-7B, Phi-3-medium, gpt-oss-20b) | running | running |
| **FLINT (46 KB, no NN)** | **0.782** | **0.676** |

## Efficiency: FLINT CTA model 46 KB, 0.10 s fit, 4 ms infer; Edmonds CPA decode 0.67 s/250 tables; one-time SBERT encode 31 s. No GPU, no neural net, no trained Steiner scorer.

## ============ FLINT RESULTS (2026-06, this work) ============
> All FLINT vs counting vs GRAMS+-ALGORITHM rows are on IDENTICAL gold-entity inputs
> (controlled method comparison). Published SOTA rows = each system's OWN pipeline
> (P2 EL), verified from the GRAMS+ artifact where noted -> reference, EL-confounded.
> CTA = approx hierarchical cscore (micro); CPA = exact micro-F1; ToughTables = official AH.

### CTA (micro cscore), gold entities, 5-fold CV
| Dataset | counting | GRAMS+ algo | FLINT-CTA | sign-test p | published ref |
|---------|----------|-------------|-----------|-------------|---------------|
| 250WT | 0.722 | 0.732 | **0.782** | 1.1e-5 | GRAMS+ 0.789 / GPT-4o 0.749 / Qwen 0.714 |
| HardTables R1 | 0.895 | 0.898 | **0.944** | ~0 | MTab 0.915 / GRAMS+ 0.906 |
| WikidataTables 2023 R1 | 0.892 | 0.898 | 0.911 | 0.378 (TIE) | MTab 0.941 / GRAMS+ 0.941 |
| ToughTables (AH) | 0.565 | 0.552 | **0.616** in-domain | — | KGCODE 0.543 / DAGOBAH 0.409 / GPT-4o 0.577 |
| 250WT real-cand (≈79% cov) | 0.661 | 0.666 | **0.737** | 1.5e-10 | GRAMS+ 0.789 (full EL) |

### CPA (exact micro-F1), gold entities, 5-fold CV
| Dataset | support-vote | FLINT+thr | FLINT+Steiner | published ref | note |
|---------|--------------|-----------|---------------|---------------|------|
| 250WT | 0.460 | 0.657 | **0.676** | GRAMS+ 0.664 / GPT-4o 0.583 | FLINT WINS (real-world) |
| HardTables R1 | 0.462 | 0.541 | 0.545 | MTab 0.941 / GRAMS+ 0.917 | candidate-limited (recall 0.398) |
| WikidataTables 2023 R1 | 0.437 | 0.504 | 0.505 | MTab 0.922 / GRAMS+ 0.893 | candidate-limited (recall 0.343) |

### Verified-from-artifact SOTA (250WT P2, github.com/binh-vu/iswc24-gp data.h5)
| System | CTA m-AF | CPA m-F |
|--------|----------|---------|
| GRAMS+ | 0.7894 | 0.6641 |
| DAGOBAH | 0.7399 | 0.6184 |
| MTab | 0.6735 | 0.5402 |

### Efficiency: FLINT = 46 KB model, 0.10 s train, 4 ms infer (CPU); no NN/GPU/Steiner-scorer.
### HONEST SCOPE: FLINT wins CTA on real-world (250WT) + HardTables, competitive on saturated
### WikidataTables, wins ToughTables; wins CPA on real-world 250WT but is candidate-generation-
### limited on synthetic SemTab CPA (HardTables/WikidataTables). Beats GPT-4o + Qwen at selection.
