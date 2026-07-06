"""Statistical significance tests for system comparisons.

Every "we beat X" claim in the paper must be backed by one of these
(CLAUDE.md s2). GRAMS+ themselves used the sign test, so reviewers expect it.

We provide:
  - paired sign test over per-table scores (matches GRAMS+ / Yeh 2000 usage)
  - paired bootstrap confidence interval on the mean score difference

Both operate on PER-TABLE scores so the unit of analysis matches the dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SignTestResult:
    n_better: int
    n_worse: int
    n_tie: int
    p_value: float

    @property
    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def paired_sign_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    tie_eps: float = 1e-9,
) -> SignTestResult:
    """Two-sided paired sign test on per-table scores (A vs B).

    Args:
        scores_a: per-table scores for our system.
        scores_b: per-table scores for the baseline, aligned table-by-table.
        tie_eps: differences with |diff| <= tie_eps count as ties.

    Returns:
        SignTestResult with the exact binomial two-sided p-value (ties excluded,
        following the standard statistical sign-test convention).
    """
    from scipy.stats import binomtest  # local import keeps utils import-light

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert a.shape == b.shape, "per-table score arrays must be aligned"
    diff = a - b
    n_better = int(np.sum(diff > tie_eps))
    n_worse = int(np.sum(diff < -tie_eps))
    n_tie = int(len(diff) - n_better - n_worse)
    n = n_better + n_worse
    if n == 0:
        return SignTestResult(n_better, n_worse, n_tie, 1.0)
    p = binomtest(n_better, n, 0.5, alternative="two-sided").pvalue
    return SignTestResult(n_better, n_worse, n_tie, float(p))


@dataclass(frozen=True)
class BootstrapCI:
    mean_diff: float
    lo: float
    hi: float
    level: float

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean per-table score difference (A - B)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = a - b
    n = len(diff)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    alpha = 1 - level
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(float(diff.mean()), float(lo), float(hi), level)
