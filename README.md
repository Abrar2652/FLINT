# FLINT: Fast Lightweight Interpretation of Tables

FLINT is a lightweight system for **Semantic Table Interpretation** — column-type
annotation (**CTA**) and column-property annotation (**CPA**) over Wikidata. It replaces
the heavy SOTA pipeline (trained neural scorers plus Steiner-tree search on a GPU) with
two tiny **gradient-boosted rankers** over closed-form ontology/statement features and a
classical **maximum-weight-arborescence** decode (Edmonds' algorithm). No neural network,
no GPU: the entire CTA model is ~46 KB and trains in seconds on CPU.

## Headline results
- **Matched entity linking (real-world 250WT):** FLINT-CPA **0.654** matches the strongest
  system GRAMS+ (0.664) and beats DAGOBAH (0.618) and MTab (0.540) — at ~1000× less compute.
- **CTA** is competitive on 250WT (ties GRAMS+ and DAGOBAH, beats MTab).
- Beats two frontier and six open (3–9B) LLMs at constrained selection over identical candidates.
- Scope: Wikidata only. Every number in the paper traces to `experiments/flint/results.json`.

## Repository layout
```
src/flint/                   the FLINT library (importable package `flint`)
  data/                        loaders, Wikidata KG cache, splits, candidate generation
  graph/                       closed-form CTA/CPA features
  eval/                        metrics + official scorers (incl. ToughTables AH)
  utils/                       paths, seeding
scripts/                     one script per experiment (flint_*, llm_*, make_*, sig_vs_sota) + data-prep
experiments/flint/           results.json (single source of truth) + per-table dumps + LLM panel
configs/default.yaml         run configuration
REPRODUCE.md                 one command per paper result
data/                        local Wikidata caches (git-ignored; build via scripts below)
legacy/                      archived earlier project (OC-GNN / HeTa-Graph) — NOT part of FLINT (git-ignored)
```
> The importable Python package is `flint` (e.g. `from flint.graph.features import ...`).

## Datasets

FLINT evaluates on the standard **SemTab** semantic-table-interpretation benchmarks (all
target **Wikidata**). The datasets are **not redistributed here** (`data/` is git-ignored) —
download them from the original sources below into `data/raw/`, then build the Wikidata
caches with `scripts/download_data.sh` and `scripts/prewarm_cache.py`. Expected table counts
per dataset are in [`docs/DATASET_FACTS.md`](docs/DATASET_FACTS.md).

| Dataset | Role | #tables | Original source |
|---|---|---|---|
| **250WT** | real-world headline eval | 250 | GRAMS+ / `sm-datasets` (`datasets/250wt`) — https://github.com/binh-vu/sm-datasets |
| **HardTables** R1 (SemTab 2022) | synthetic; has CPA GT | 3,891 | Zenodo — https://zenodo.org/records/7416036 |
| **WikidataTables** 2023 R1 (SemTab 2023) | synthetic | 10,417 | Zenodo — https://zenodo.org/records/8393535 |
| **ToughTables / 2T_WD** (SemTab) | CTA robustness (no CPA GT) | 180 | Zenodo — https://zenodo.org/records/4246370 |
| **tFood** (SemTab 2023) | domain-specific corroboration | — | Zenodo — https://zenodo.org/records/10048187 |



## Quickstart
```bash
pip install -e ".[dev]"
python scripts/prewarm_cache.py --dataset 250wt        # warm the Wikidata cache (entities, types, P279 closure)
python scripts/flint_cta_ranker.py --dataset 250wt     # CTA
python scripts/flint_cpa_ranker.py --dataset 250wt     # CPA (Steiner/arborescence decode)
```
See **`REPRODUCE.md`** for one command per result and the exact protocols (metrics, gold vs.
real entity linking, folds). LLM baselines read API keys from `.env` at runtime and never commit them.

## Requirements
Python ≥ 3.10; deps in `pyproject.toml` (`pip install -e ".[dev]"`, add `".[llm]"` for the API LLM
baselines). A frozen SBERT pass provides label embeddings; there is no model training beyond the
gradient-boosted rankers.
