"""Tests for significance utilities (implemented now -> must pass)."""
from flint.eval.significance import paired_sign_test, paired_bootstrap_ci


def test_sign_test_clear_win():
    a = [0.9, 0.8, 0.85, 0.88, 0.92, 0.79, 0.81]
    b = [0.7, 0.6, 0.65, 0.62, 0.71, 0.59, 0.61]
    res = paired_sign_test(a, b)
    assert res.n_better == 7 and res.n_worse == 0
    assert res.p_value < 0.05


def test_sign_test_ties_excluded():
    a = [0.5, 0.5, 0.9]
    b = [0.5, 0.5, 0.1]
    res = paired_sign_test(a, b)
    assert res.n_tie == 2 and res.n_better == 1 and res.n_worse == 0


def test_bootstrap_ci_excludes_zero_on_clear_win():
    a = [0.9, 0.8, 0.85, 0.88, 0.92]
    b = [0.7, 0.6, 0.65, 0.62, 0.71]
    ci = paired_bootstrap_ci(a, b, n_resamples=2000, seed=0)
    assert ci.mean_diff > 0 and ci.excludes_zero
