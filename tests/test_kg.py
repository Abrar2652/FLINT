"""Tests for the KG ontology layer (src/flint/data/kg.py).

Uses the in-memory FakeKG so the suite is fully offline/deterministic. A tiny
hierarchy mirrors the real Q5(human) -> Q215627(person) chain used by cscore.
"""
from __future__ import annotations

from flint.data.kg import (
    FakeKG,
    ancestors_with_hops,
    hierarchy_distance,
    is_ancestor,
)
from flint.eval.metrics import cscore

# human -> person -> being ; cat -> being ; with a cycle guard case
KG = FakeKG(
    instance_of={"Qdan": ["Q5"]},
    subclass_of={
        "Q5": ["Q215627"],        # human subclass-of person
        "Q215627": ["Qbeing"],    # person subclass-of being
        "Q146": ["Qbeing"],       # cat subclass-of being
        "Qcycle1": ["Qcycle2"],   # cycle: should not hang
        "Qcycle2": ["Qcycle1"],
    },
    statements={"Qdan": [("P54", "Qteam")]},
    labels={"Q5": "human", "Q215627": "person"},
)


def test_ancestors_with_hops():
    anc = ancestors_with_hops(KG, "Q5")
    assert anc == {"Q5": 0, "Q215627": 1, "Qbeing": 2}


def test_ancestors_respects_max_hops():
    assert ancestors_with_hops(KG, "Q5", max_hops=1) == {"Q5": 0, "Q215627": 1}


def test_cycle_does_not_hang():
    anc = ancestors_with_hops(KG, "Qcycle1")
    assert anc == {"Qcycle1": 0, "Qcycle2": 1}


def test_hierarchy_distance_and_ancestor():
    assert hierarchy_distance(KG, "Q5", "Q215627") == 1
    assert hierarchy_distance(KG, "Q5", "Qbeing") == 2
    assert hierarchy_distance(KG, "Q5", "Q146") is None  # siblings: unrelated
    assert is_ancestor(KG, "Q215627", "Q5") is True
    assert is_ancestor(KG, "Q5", "Q215627") is False


def test_cscore_with_kg_matches_gramsplus_scheme():
    hd = lambda a, b: hierarchy_distance(KG, a, b)  # noqa: E731
    isa = lambda a, b: is_ancestor(KG, a, b)        # noqa: E731
    assert cscore("Q5", "Q5", hierarchy_distance=hd, is_ancestor=isa) == 1.0
    # predict person (too general) for gold human: 0.8**1
    assert round(cscore("Q215627", "Q5", hierarchy_distance=hd, is_ancestor=isa), 4) == 0.8
    # predict human (too specific) for gold person: 0.7**1
    assert round(cscore("Q5", "Q215627", hierarchy_distance=hd, is_ancestor=isa), 4) == 0.7
    # unrelated
    assert cscore("Q5", "Q146", hierarchy_distance=hd, is_ancestor=isa) == 0.0


def test_fake_kg_protocol_surface():
    assert KG.instance_of("Qdan") == ["Q5"]
    assert KG.statements("Qdan") == [("P54", "Qteam")]
    assert KG.label("Q5") == "human"
