"""Tests for the aggregate eval (eval/evaluate.py) and frozen splits (data/splits.py)."""
from __future__ import annotations

from flint.data.kg import FakeKG
from flint.data.splits import make_kfold
from flint.eval.evaluate import evaluate_cpa, evaluate_cta

KG = FakeKG(subclass_of={"Q5": ["Q215627"], "Q215627": ["Qbeing"]})


def test_kfold_is_deterministic_and_partitions():
    ids = [f"t{i}" for i in range(23)]
    folds = make_kfold(ids, k=5, seed=0)
    assert make_kfold(ids, k=5, seed=0) == folds          # deterministic
    flat = [x for f in folds for x in f]
    assert sorted(flat) == sorted(ids)                    # partition (no loss/dup)
    assert len(set(flat)) == len(ids)
    assert make_kfold(ids, k=5, seed=1) != folds          # seed changes it


def test_evaluate_cta_exact_and_partial_credit():
    # t1 col0: exact (1.0); t2 col0: predicted ancestor person/human (0.8);
    # t2 col1: predicted Q5 vs unrelated Qzzz (0.0, no hierarchy path)
    pred = {"t1": {0: "Q5"}, "t2": {0: "Q215627", 1: "Q5"}}
    gold = {"t1": {0: "Q5"}, "t2": {0: "Q5", 1: "Qzzz"}}
    m = evaluate_cta(pred, gold, KG)
    # micro: cscores = [1.0, 0.8, 0.0] over 3 predicted and 3 gold -> P=R=1.8/3=0.6
    assert abs(m["cta_micro_p"] - 0.6) < 1e-6
    assert abs(m["cta_micro_r"] - 0.6) < 1e-6
    assert abs(m["cta_micro_f1"] - 0.6) < 1e-6


def test_evaluate_cta_abstention_lowers_recall_not_precision():
    # predict only the correct one; abstain on the other -> P=1.0, R=0.5
    pred = {"t1": {0: "Q5"}}
    gold = {"t1": {0: "Q5", 1: "Qbeing"}}
    m = evaluate_cta(pred, gold, KG)
    assert abs(m["cta_micro_p"] - 1.0) < 1e-6
    assert abs(m["cta_micro_r"] - 0.5) < 1e-6


def test_evaluate_cpa_exact_micro():
    pred = {"t1": {(0, 1): "P54", (0, 2): "P999"}}   # one right, one wrong
    gold = {"t1": {(0, 1): "P54", (0, 2): "P1351"}}
    m = evaluate_cpa(pred, gold)
    assert abs(m["cpa_micro_p"] - 0.5) < 1e-6
    assert abs(m["cpa_micro_r"] - 0.5) < 1e-6
    assert abs(m["cpa_micro_f1"] - 0.5) < 1e-6
