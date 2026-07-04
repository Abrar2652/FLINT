"""Gate-1 tests for the 250WT loader (src/flint/data/loaders.py).

Skips cleanly when the `sm` (sem-desc) dependency or the raw 250WT data are
absent, so the suite still runs on a bare checkout.
"""
from __future__ import annotations

import pytest

from flint.utils.paths import dataset_dir

pytest.importorskip("sm.dataset", reason="sem-desc not installed")

if not (dataset_dir("250wt") / "tables").exists():
    pytest.skip("250WT raw data not downloaded", allow_module_level=True)

from flint.data.loaders import STIRecord, _wd_id, load_dataset  # noqa: E402


@pytest.fixture(scope="module")
def wt250() -> list[STIRecord]:
    return list(load_dataset("250wt"))


def test_wd_id_normalisation():
    assert _wd_id("http://www.wikidata.org/entity/Q5") == "Q5"
    assert _wd_id("http://www.wikidata.org/prop/P102") == "P102"
    assert _wd_id("http://www.wikidata.org/prop/direct/P31") == "P31"
    assert _wd_id("http://www.w3.org/2000/01/rdf-schema#label") is None


def test_loads_all_250_tables(wt250):
    assert len(wt250) == 250
    assert len({r.table_id for r in wt250}) == 250


def test_hand_verified_example(wt250):
    """11th_Lok_Sabha was decoded by hand from the raw semantic model."""
    r = next(r for r in wt250 if r.table_id == "11th_Lok_Sabha")
    # CTA: human / electoral-constituency / political-party
    assert r.gold_cta == {1: "Q47481352", 3: "Q5", 4: "Q7278"}
    # CPA: member-of-party (direct) and electoral-district (reified via P39 statement)
    assert r.gold_cpa[(3, 4)] == "P102"
    assert r.gold_cpa[(3, 1)] == "P768"
    assert r.cpa_via_statement[(3, 1)]["statement_property"] == "P39"
    # CEA: links are present and look like Q-ids
    assert r.gold_cea[(0, 1)] == "Q3883318"
    assert all(v.startswith("Q") for v in r.gold_cea.values())


def test_corpus_invariants(wt250):
    # every table has at least one CTA label and CEA links
    assert all(r.gold_cta for r in wt250)
    assert all(r.gold_cea for r in wt250)
    # rows are rectangular: every row has len == #headers
    for r in wt250:
        assert all(len(row) == len(r.headers) for row in r.rows)
    # entity_columns are valid indices and a superset of CTA columns
    for r in wt250:
        assert set(r.gold_cta) <= set(r.entity_columns)
        assert all(0 <= c < len(r.headers) for c in r.entity_columns)
    # distinct-class count matches the paper's ~150 (sanity on extraction)
    classes = {c for r in wt250 for c in r.gold_cta.values()}
    assert 140 <= len(classes) <= 170


def test_cpa_indices_are_distinct_columns(wt250):
    for r in wt250:
        for (i, j), p in r.gold_cpa.items():
            assert i != j
            assert 0 <= i < len(r.headers) and 0 <= j < len(r.headers)
            assert p.startswith("P")
