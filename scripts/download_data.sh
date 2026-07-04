#!/usr/bin/env bash
# Gate 0: acquire datasets. Verifies row counts against docs/DATASET_FACTS.md.
set -euo pipefail
RAW=data/raw
mkdir -p "$RAW"
echo "[download] datasets -> $RAW"
echo "  250WT:            clone github.com/usc-isi-i2/GRAMS (tag iswc-2021), copy its 250WT data"
echo "  HardTables R1 22: https://zenodo.org/records/7416036"
echo "  WikidataTbls 23:  https://zenodo.org/records/8393535"
echo "  ToughTables/2T:   https://zenodo.org/records/4246370"
echo "  tFood:            SemTab 2023 distribution"
echo
echo "[Gate 0 ACTION] Verify https://purl.org/gramsplus resolves to USABLE code/models."
echo "  -> record the result in docs/GATE_STATUS.md (decides baseline strategy, CLAUDE.md s4)."
echo
echo "Automated fetch is intentionally NOT wired up: Zenodo/repo URLs and licenses"
echo "must be confirmed by a human first. Fill in curl/wget calls here once confirmed,"
echo "then run scripts/verify_data.py to check counts against docs/DATASET_FACTS.md."
