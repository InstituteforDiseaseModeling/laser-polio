<<<<<<< HEAD
import numpy as np
import time
import pytest
from laser_core.propertyset import PropertySet
from laser_polio.model import populate_heterogeneous_values, populate_heterogeneous_values_slow

def _make_pars(seed=None):
    return PropertySet({
        "r0": 12.0,
        "risk_mult_var": 0.75,
        "corr_risk_inf": 0.4,
        "individual_heterogeneity": True,
        "seed": seed,
        "dur_inf": lambda n: np.full(n, 10.0, dtype=float),
    })

#@pytest.mark.slow
def test_hetero_population_runtime_comparison():
    N = 10_000_000  # adjust as needed
=======
import time
import warnings

import numpy as np
import scipy.stats as stats
from laser_core.propertyset import PropertySet

from laser_polio.model import populate_heterogeneous_values


def populate_heterogeneous_values_reference_slow(start, end, acq_risk_out, infectivity_out, pars):
    """Trusted slow reference version of heterogeneity kernel."""
    mean_ln = 1
    var_ln = pars.risk_mult_var
    mu_ln = np.log(mean_ln**2 / np.sqrt(var_ln + mean_ln**2))
    sigma_ln = np.sqrt(np.log(var_ln / mean_ln**2 + 1))
    mean_gamma = pars.r0 / np.mean(pars.dur_inf(1000))
    shape_gamma = 1
    scale_gamma = max(mean_gamma / shape_gamma, 1e-10)

    rho = pars.corr_risk_inf
    cov_matrix = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(cov_matrix)

    BATCH_SIZE = 1_000_000
    for batch_start in range(start, end, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, end)
        b_n = batch_end - batch_start

        z = np.random.normal(size=(b_n, 2))
        z_corr = z @ L.T

        if pars.individual_heterogeneity:
            acq_risk_out[batch_start:batch_end] = np.exp(mu_ln + sigma_ln * z_corr[:, 0])
            infectivity_out[batch_start:batch_end] = stats.gamma.ppf(stats.norm.cdf(z_corr[:, 1]), a=shape_gamma, scale=scale_gamma)
        else:
            acq_risk_out[batch_start:batch_end] = 1.0
            infectivity_out[batch_start:batch_end] = mean_gamma


def _make_pars(seed=None):
    return PropertySet(
        {
            "r0": 12.0,
            "risk_mult_var": 0.75,
            "corr_risk_inf": 0.4,
            "individual_heterogeneity": True,
            "seed": seed,
            "dur_inf": lambda n: np.full(n, 10.0, dtype=float),
        }
    )


def test_hetero_population_runtime_comparison():
    N = 10_000_000
>>>>>>> fix_init_hotspot2
    acq_fast = np.empty(N, dtype=np.float32)
    inf_fast = np.empty(N, dtype=np.float32)
    acq_slow = np.empty(N, dtype=np.float32)
    inf_slow = np.empty(N, dtype=np.float32)
    pars = _make_pars(seed=42)

<<<<<<< HEAD
    t0 = time.perf_counter()
    populate_heterogeneous_values_slow(0, N, acq_slow, inf_slow, pars)
    t1 = time.perf_counter()

=======
    # Run slow reference version
    t0 = time.perf_counter()
    populate_heterogeneous_values_reference_slow(0, N, acq_slow, inf_slow, pars)
    t1 = time.perf_counter()

    # Run fast tested version
>>>>>>> fix_init_hotspot2
    populate_heterogeneous_values(0, N, acq_fast, inf_fast, pars)
    t2 = time.perf_counter()

    time_slow = t1 - t0
    time_fast = t2 - t1
    speedup = time_slow / time_fast if time_fast > 0 else float("inf")
<<<<<<< HEAD

    print(f"\nSlow: {time_slow:.3f}s | Fast: {time_fast:.3f}s | Speedup: {speedup:.2f}x")

    # Validate that results are not wildly off
    assert np.allclose(acq_fast.mean(), 1.0, rtol=0.01)
    assert np.allclose(inf_fast.mean(), pars.r0 / 10.0, rtol=0.01)

    # Optional: alert if speedup is below expectation
    if speedup < 4.0:
        pytest.warns(UserWarning, f"Speedup lower than expected: {speedup:.2f}x")


#@pytest.mark.slow
=======
    print(f"\n⏱️ Slow: {time_slow:.3f}s | Fast: {time_fast:.3f}s | Speedup: {speedup:.2f}x")

    # Compare standard deviation
    np.testing.assert_allclose(acq_fast.std(), acq_slow.std(), rtol=0.01)
    np.testing.assert_allclose(inf_fast.std(), inf_slow.std(), rtol=0.01)

    # Compare percentiles
    for p in [10, 50, 90]:
        np.testing.assert_allclose(np.percentile(acq_fast, p), np.percentile(acq_slow, p), rtol=0.02)
        np.testing.assert_allclose(np.percentile(inf_fast, p), np.percentile(inf_slow, p), rtol=0.02)

    # Compare copula: correlation
    corr_slow = np.corrcoef(acq_slow, inf_slow)[0, 1]
    corr_fast = np.corrcoef(acq_fast, inf_fast)[0, 1]
    np.testing.assert_allclose(corr_fast, corr_slow, atol=0.02)

    # Optional: warn if speedup too low
    if speedup < 4.0:
        warnings.warn(f"Speedup lower than expected: {speedup:.2f}x", UserWarning, stacklevel=2)


>>>>>>> fix_init_hotspot2
def test_large_population_no_crash():
    N = 100_000_000
    acq = np.empty(N, dtype=np.float32)
    inf = np.empty(N, dtype=np.float32)
    pars = _make_pars(seed=123)

    populate_heterogeneous_values(0, N, acq, inf, pars)

    assert np.isfinite(acq).all()
    assert np.isfinite(inf).all()
