"""Tests for the metrics module (implemented now -> these must pass)."""
from flint.eval.metrics import (
    PRF, micro_prf, macro_prf, cscore, coverage, average_set_size,
)


def test_micro_prf_perfect():
    r = micro_prf({"c0": "P54", "c1": "P27"}, {"c0": "P54", "c1": "P27"})
    assert r == PRF(1.0, 1.0, 1.0)


def test_micro_prf_partial():
    # 1 correct of 2 predicted; gold has 2 -> P=0.5, R=0.5, F1=0.5
    r = micro_prf({"c0": "P54", "c1": "PWRONG"}, {"c0": "P54", "c1": "P27"})
    assert abs(r.precision - 0.5) < 1e-9
    assert abs(r.recall - 0.5) < 1e-9
    assert abs(r.f1 - 0.5) < 1e-9


def test_micro_prf_abstention_helps_precision():
    # abstaining on the hard column (omit it) -> P=1.0, R=0.5
    r = micro_prf({"c0": "P54"}, {"c0": "P54", "c1": "P27"})
    assert abs(r.precision - 1.0) < 1e-9
    assert abs(r.recall - 0.5) < 1e-9


def test_macro_is_mean_of_tables():
    t1_pred, t1_gold = {"a": 1}, {"a": 1}            # F1 = 1.0
    t2_pred, t2_gold = {"a": 9}, {"a": 1}            # F1 = 0.0
    r = macro_prf([t1_pred, t2_pred], [t1_gold, t2_gold])
    assert abs(r.f1 - 0.5) < 1e-9


def test_cscore_exact_and_ancestor_and_descendant():
    # toy 2-level hierarchy: public_uni -< university -< edu_institution
    parents = {"public_uni": "university", "university": "edu_institution"}

    def dist(a, b):
        # hops along parent chain in either direction, else None
        for src, dst in ((a, b), (b, a)):
            cur, d = src, 0
            while cur is not None:
                if cur == dst:
                    return d
                cur = parents.get(cur)
                d += 1
        return None

    def is_anc(a, b):
        cur = parents.get(b)
        while cur is not None:
            if cur == a:
                return True
            cur = parents.get(cur)
        return False

    assert cscore("university", "university", hierarchy_distance=dist, is_ancestor=is_anc) == 1.0
    # predicted ancestor at distance 1 -> 0.8
    assert abs(cscore("university", "public_uni", hierarchy_distance=dist, is_ancestor=is_anc) - 0.8) < 1e-9
    # predicted descendant at distance 1 -> 0.7
    assert abs(cscore("public_uni", "university", hierarchy_distance=dist, is_ancestor=is_anc) - 0.7) < 1e-9


def test_coverage_and_set_size():
    sets = [{1, 2}, {3}, {4, 5, 6}]
    golds = [1, 3, 99]  # third is a miss
    assert abs(coverage(sets, golds) - (2 / 3)) < 1e-9
    assert abs(average_set_size(sets) - 2.0) < 1e-9
