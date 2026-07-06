"""Evaluation metrics for STI.

This is the ONLY module allowed to compute CTA/CPA/CEA scores (CLAUDE.md s2).
Two metric families, never to be mixed within a single results table:

  - CTA  -> SemTab approximate hierarchical `cscore` (partial credit by hierarchy distance)
  - CEA/CPA -> exact micro precision / recall / F1

We expose both micro and macro variants so we can match whatever a baseline reported.

The cscore implementation follows the GRAMS+ description:
    score(x) = 0.8 ** d(x)   if d(x) <= 5 and x is correct or an ancestor of GT
    score(x) = 0.7 ** d(x)   if d(x) <= 3 and x is a descendant of GT
    score(x) = 0            otherwise
where d(x) is the shortest hierarchy distance from prediction x to the ground-truth item.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PRF:
    """Precision / recall / F1 triple."""

    precision: float
    recall: float
    f1: float


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def micro_prf(
    predictions: Mapping[Hashable, Hashable],
    gold: Mapping[Hashable, Hashable],
) -> PRF:
    """Exact micro P/R/F1 for CEA/CPA.

    Args:
        predictions: mapping target_id -> predicted_item. Targets with no
            prediction should be ABSENT (abstention), not mapped to None.
        gold: mapping target_id -> gold_item over the annotated targets.

    Returns:
        PRF computed as: precision = correct / predicted,
        recall = correct / gold, exact item match.
    """
    predicted = set(predictions)
    correct = sum(
        1 for t, p in predictions.items() if t in gold and p == gold[t]
    )
    p = correct / len(predicted) if predicted else 0.0
    r = correct / len(gold) if gold else 0.0
    return PRF(p, r, _f1(p, r))


def macro_prf(
    predictions_per_table: Sequence[Mapping[Hashable, Hashable]],
    gold_per_table: Sequence[Mapping[Hashable, Hashable]],
) -> PRF:
    """Macro P/R/F1 = unweighted mean of per-table micro PRF.

    Used to match GRAMS+ Table 1 (macro-average approximate) reporting.
    """
    assert len(predictions_per_table) == len(gold_per_table)
    if not gold_per_table:
        return PRF(0.0, 0.0, 0.0)
    triples = [
        micro_prf(pred, gold)
        for pred, gold in zip(predictions_per_table, gold_per_table)
    ]
    n = len(triples)
    p = sum(t.precision for t in triples) / n
    r = sum(t.recall for t in triples) / n
    f = sum(t.f1 for t in triples) / n
    return PRF(p, r, f)


def cscore(
    pred_item: Hashable,
    gold_item: Hashable,
    *,
    hierarchy_distance: Callable[[Hashable, Hashable], int | None],
    is_ancestor: Callable[[Hashable, Hashable], bool],
) -> float:
    """SemTab approximate hierarchical score for one CTA prediction.

    Args:
        pred_item: predicted class.
        gold_item: ground-truth class.
        hierarchy_distance: returns the shortest |hops| between two classes in the
            ontology, or None if they are not on the same ancestor/descendant path.
        is_ancestor: is_ancestor(a, b) is True iff a is an ancestor of b.

    Returns:
        Approximate score in [0, 1] per the GRAMS+ scoring scheme.
    """
    if pred_item == gold_item:
        return 1.0
    d = hierarchy_distance(pred_item, gold_item)
    if d is None:
        return 0.0
    # ancestor of GT (predicted too general): 0.8**d up to distance 5
    if is_ancestor(pred_item, gold_item):
        return 0.8 ** d if d <= 5 else 0.0
    # descendant of GT (predicted too specific): 0.7**d up to distance 3
    if is_ancestor(gold_item, pred_item):
        return 0.7 ** d if d <= 3 else 0.0
    return 0.0


def cta_micro_approx(
    predictions: Mapping[Hashable, Hashable],
    gold: Mapping[Hashable, Hashable],
    *,
    hierarchy_distance: Callable[[Hashable, Hashable], int | None],
    is_ancestor: Callable[[Hashable, Hashable], bool],
) -> PRF:
    """Micro approximate-F1 for CTA using cscore as partial-credit numerator.

    Precision numerator and recall numerator both use the summed cscore over
    predicted columns; denominators are #predicted and #gold respectively.
    """
    scored = {
        t: cscore(
            p,
            gold[t],
            hierarchy_distance=hierarchy_distance,
            is_ancestor=is_ancestor,
        )
        for t, p in predictions.items()
        if t in gold
    }
    total = sum(scored.values())
    p = total / len(predictions) if predictions else 0.0
    r = total / len(gold) if gold else 0.0
    return PRF(p, r, _f1(p, r))


def coverage(
    prediction_sets: Iterable[set],
    gold_items: Iterable[Hashable],
) -> float:
    """Empirical marginal coverage of conformal prediction sets.

    coverage = fraction of targets whose gold item is inside its prediction set.
    The conformal guarantee says this should be >= 1 - alpha (CLAUDE.md Gate 4).
    """
    sets = list(prediction_sets)
    golds = list(gold_items)
    assert len(sets) == len(golds)
    if not golds:
        return 0.0
    hit = sum(1 for s, g in zip(sets, golds) if g in s)
    return hit / len(golds)


def average_set_size(prediction_sets: Iterable[set]) -> float:
    """Mean cardinality of conformal prediction sets (efficiency metric)."""
    sets = list(prediction_sets)
    return sum(len(s) for s in sets) / len(sets) if sets else 0.0
