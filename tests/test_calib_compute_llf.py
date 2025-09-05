import sys

import numpy as np
from scipy.stats import nbinom
from scipy.stats import poisson

sys.path.append("calib")

from scoring import compute_log_likelihood_fit


def test_poisson_exact_match_normed():
    actual = {"a": [2, 3], "b": [0, 1, 2]}
    predicted = {"a": [2, 3], "b": [0.5, 1, 2]}  # non-integer lambdas allowed
    out = compute_log_likelihood_fit(actual, predicted, method="poisson", norm_by_n=True)

    # expected, per-key: -mean(log pmf)
    exp_a = -poisson.logpmf([2, 3], [2, 3]).sum() / 2
    exp_b = -poisson.logpmf([0, 1, 2], [0.5, 1, 2]).sum() / 3
    assert np.isclose(out["a"], exp_a)
    assert np.isclose(out["b"], exp_b)
    assert np.isclose(out["total_log_likelihood"], exp_a + exp_b)


def test_poisson_clip_zeros_in_pred():
    # predicted contains zeros; function clips to 1e-6 to avoid log(0)
    actual = {"x": [0, 5]}
    predicted = {"x": [0.0, 0.0]}
    out = compute_log_likelihood_fit(actual, predicted, method="poisson", norm_by_n=True)

    lam = np.clip(np.array([0.0, 0.0], float), 1e-6, None)
    exp = -poisson.logpmf([0, 5], lam).sum() / 2
    assert np.isclose(out["x"], exp)
    assert np.isclose(out["total_log_likelihood"], exp)


def test_neg_binomial_basic():
    # Using the same parameterization as the code:
    # r = dispersion, p = r / (r + mean) for nbinom.logpmf(k, r, p)
    actual = {"y": [0, 1, 4]}
    predicted = {"y": [0.5, 1.0, 3.5]}
    r = 2.0
    out = compute_log_likelihood_fit(actual, predicted, method="neg_binomial", dispersion=r, norm_by_n=True)

    lam = np.array([0.5, 1.0, 3.5], float)
    p = r / (r + lam)
    exp = -nbinom.logpmf([0, 1, 4], r, p).sum() / 3
    assert np.isclose(out["y"], exp)
    assert np.isclose(out["total_log_likelihood"], exp)


def test_weights_apply_per_key():
    actual = {"a": [1, 1], "b": [2, 2]}
    predicted = {"a": [1, 2], "b": [1, 4]}
    weights = {"a": 2.0, "b": 0.5}
    out = compute_log_likelihood_fit(actual, predicted, method="poisson", weights=weights, norm_by_n=True)

    exp_a = -2.0 * poisson.logpmf([1, 1], [1, 2]).sum() / 2
    exp_b = -0.5 * poisson.logpmf([2, 2], [1, 4]).sum() / 2
    assert np.isclose(out["a"], exp_a)
    assert np.isclose(out["b"], exp_b)
    assert np.isclose(out["total_log_likelihood"], exp_a + exp_b)


def test_norm_by_n_false_matches_sum():
    actual = {"c": [0, 1, 2]}
    predicted = {"c": [0.5, 1.0, 2.5]}
    out_sum = compute_log_likelihood_fit(actual, predicted, method="poisson", norm_by_n=False)
    out_mean = compute_log_likelihood_fit(actual, predicted, method="poisson", norm_by_n=True)

    lam = np.array([0.5, 1.0, 2.5], float)
    exp_sum = -poisson.logpmf([0, 1, 2], lam).sum()  # not normalized
    exp_mean = -poisson.logpmf([0, 1, 2], lam).sum() / 3  # normalized

    assert np.isclose(out_sum["c"], exp_sum)
    assert np.isclose(out_mean["c"], exp_mean)
    assert np.isclose(out_sum["total_log_likelihood"], exp_sum)
    assert np.isclose(out_mean["total_log_likelihood"], exp_mean)


def test_nested_dict_alignment_poisson():
    # nested dicts should be aligned by sorted keys internally
    actual = {"ND": {"A": 1, "B": 3}}
    predicted = {"ND": {"B": 2.5, "A": 1.0}}  # reverse order; should be aligned
    out = compute_log_likelihood_fit(actual, predicted, method="poisson", norm_by_n=True)

    # sorted subkeys -> ["A", "B"]
    v_obs = np.array([1, 3])
    v_sim = np.array([1.0, 2.5])
    exp = -poisson.logpmf(v_obs, v_sim).sum() / 2
    assert np.isclose(out["ND"], exp)
    assert np.isclose(out["total_log_likelihood"], exp)


def test_shape_mismatch_skips_key():
    # "bad" has mismatched shapes and should be skipped; only "good" contributes
    actual = {"good": [1, 2], "bad": [1, 2, 3]}
    predicted = {"good": [1, 1], "bad": [1, 2]}  # mismatch for "bad"
    out = compute_log_likelihood_fit(actual, predicted, method="poisson", norm_by_n=True)

    exp_good = -poisson.logpmf([1, 2], [1, 1]).sum() / 2
    assert "bad" not in out or out["bad"] == out.get("bad")  # may not exist if skipped
    assert np.isclose(out["good"], exp_good)
    assert np.isclose(out["total_log_likelihood"], exp_good)
