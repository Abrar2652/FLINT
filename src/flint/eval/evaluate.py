"""Aggregate CTA/CPA evaluation over many tables, wiring metrics.py to the KG.

CTA -> SemTab approximate hierarchical cscore (micro + macro).
CPA -> exact micro / macro P/R/F1.
These are the only numbers that go in a results table (CLAUDE.md s2); never mix
the two families. Predictions/gold are per-table dicts so both micro (pool all
targets) and macro (mean over tables) are available.
"""
from __future__ import annotations

from flint.data.kg import KG, hierarchy_distance, is_ancestor
from flint.eval.metrics import PRF, _f1, cscore, cta_micro_approx, macro_prf, micro_prf


def evaluate_cta(
    pred_per_table: dict[str, dict[int, str]],
    gold_per_table: dict[str, dict[int, str]],
    kg: KG,
) -> dict[str, float]:
    """Approximate hierarchical CTA score (cscore). Returns micro & macro P/R/F1."""
    hd = lambda a, b: hierarchy_distance(kg, a, b)   # noqa: E731
    isa = lambda a, b: is_ancestor(kg, a, b)         # noqa: E731

    # micro: pool every (table, col) target into one mapping
    micro_pred: dict[tuple[str, int], str] = {}
    micro_gold: dict[tuple[str, int], str] = {}
    for tid, gold in gold_per_table.items():
        pred = pred_per_table.get(tid, {})
        for col, g in gold.items():
            micro_gold[(tid, col)] = g
            if col in pred:
                micro_pred[(tid, col)] = pred[col]
    micro = cta_micro_approx(micro_pred, micro_gold, hierarchy_distance=hd, is_ancestor=isa)

    # macro: per-table cscore then unweighted mean
    per_table = [
        cta_micro_approx(
            {c: p for c, p in pred_per_table.get(tid, {}).items()},
            gold,
            hierarchy_distance=hd,
            is_ancestor=isa,
        )
        for tid, gold in gold_per_table.items()
        if gold
    ]
    n = len(per_table) or 1
    macro = PRF(
        sum(t.precision for t in per_table) / n,
        sum(t.recall for t in per_table) / n,
        sum(t.f1 for t in per_table) / n,
    )
    return {
        "cta_micro_f1": micro.f1, "cta_micro_p": micro.precision, "cta_micro_r": micro.recall,
        "cta_macro_f1": macro.f1, "cta_macro_p": macro.precision, "cta_macro_r": macro.recall,
    }


def evaluate_cta_multigold(
    pred_per_table: dict[str, dict[int, str]],
    accept_per_table: dict[str, dict[int, set]],
    kg: KG,
) -> dict[str, float]:
    """Multi-valid-type CTA cscore (ToughTables/tFood): a column's gold is a SET of
    acceptable classes; score = 1.0 if pred is in the set, else the best partial
    (ancestor/descendant) cscore against any member. Returns micro & macro F1."""
    hd = lambda a, b: hierarchy_distance(kg, a, b)   # noqa: E731
    isa = lambda a, b: is_ancestor(kg, a, b)         # noqa: E731

    def col_score(pred: str, accept: set) -> float:
        if pred in accept:
            return 1.0
        best = 0.0
        for g in accept:
            best = max(best, cscore(pred, g, hierarchy_distance=hd, is_ancestor=isa))
        return best

    # micro: pool every (table,col) target
    num = den_pred = den_gold = 0.0
    per_table = []
    for tid, accept in accept_per_table.items():
        pred = pred_per_table.get(tid, {})
        t_num = t_pred = t_gold = 0.0
        for col, acc in accept.items():
            if not acc:
                continue
            den_gold += 1
            t_gold += 1
            if col in pred:
                s = col_score(pred[col], acc)
                num += s
                den_pred += 1
                t_num += s
                t_pred += 1
        tp = t_num / t_pred if t_pred else 0.0
        tr = t_num / t_gold if t_gold else 0.0
        per_table.append(_f1(tp, tr))
    mp = num / den_pred if den_pred else 0.0
    mr = num / den_gold if den_gold else 0.0
    macro = sum(per_table) / len(per_table) if per_table else 0.0
    return {"cta_micro_f1": _f1(mp, mr), "cta_micro_p": mp, "cta_micro_r": mr,
            "cta_macro_f1": macro}


def evaluate_cpa(
    pred_per_table: dict[str, dict[tuple[int, int], str]],
    gold_per_table: dict[str, dict[tuple[int, int], str]],
) -> dict[str, float]:
    """Exact micro & macro P/R/F1 for CPA."""
    # micro: pool all (table, i, j) targets
    micro_pred: dict[tuple, str] = {}
    micro_gold: dict[tuple, str] = {}
    for tid, gold in gold_per_table.items():
        pred = pred_per_table.get(tid, {})
        for k, g in gold.items():
            micro_gold[(tid, *k)] = g
        for k, p in pred.items():
            micro_pred[(tid, *k)] = p
    micro = micro_prf(micro_pred, micro_gold)

    macro = macro_prf(
        [pred_per_table.get(tid, {}) for tid in gold_per_table],
        [gold_per_table[tid] for tid in gold_per_table],
    )
    return {
        "cpa_micro_f1": micro.f1, "cpa_micro_p": micro.precision, "cpa_micro_r": micro.recall,
        "cpa_macro_f1": macro.f1, "cpa_macro_p": macro.precision, "cpa_macro_r": macro.recall,
    }
