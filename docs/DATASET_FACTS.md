# DATASET_FACTS.md — expected dataset shapes (Gate 0 checks against these)

| Dataset | #tables (est) | #tables (ACTUAL, acquired 2026-06-05) | Source | Target KG | Notes |
|---------|---------|---------|--------|-----------|-------|
| 250WT | 250 | **250** (+250 GT) | binh-vu/sm-datasets `datasets/250wt` v150 | Wikidata | real Wikipedia tables; HEADLINE eval |
| HardTables R1 2022 | 3,650 | **3,891** (cea+cpa+cta GT) | Zenodo 7416036 | Wikidata | synthetic, near-saturated; HAS CPA GT |
| WikidataTables 2023 R1 | 9,920 | **10,417** (Valid/Test) | Zenodo 8393535 | Wikidata | synthetic |
| ToughTables / 2T_WD | ~180 | **180** (CEA+CTA GT, NO CPA GT) | Zenodo 4246370 | Wikidata | noisy/ambiguous; GT disputed; CTA-robustness only |
| tFood | — | **entity+horizontal, test/val** (cta+cpa+cea GT) | Zenodo 10048187 | Wikidata | domain-specific, very hard; HAS CPA GT; Gate-5 corroboration |

Training data: ~5,000 auto-labeled Wikipedia tables (distant supervision).
LEAKAGE GUARD: exclude any Wikipedia article whose table appears in 250WT.
Snapshots used by GRAMS+: Wikidata dump 2023-06-19, Wikipedia dump 2023-06-20.
