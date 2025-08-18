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
    acq_fast = np.empty(N, dtype=np.float32)
    inf_fast = np.empty(N, dtype=np.float32)
    acq_slow = np.empty(N, dtype=np.float32)
    inf_slow = np.empty(N, dtype=np.float32)
    pars = _make_pars(seed=42)

    t0 = time.perf_counter()
    populate_heterogeneous_values_slow(0, N, acq_slow, inf_slow, pars)
    t1 = time.perf_counter()

    populate_heterogeneous_values(0, N, acq_fast, inf_fast, pars)
    t2 = time.perf_counter()

    time_slow = t1 - t0
    time_fast = t2 - t1
    speedup = time_slow / time_fast if time_fast > 0 else float("inf")

    print(f"\nSlow: {time_slow:.3f}s | Fast: {time_fast:.3f}s | Speedup: {speedup:.2f}x")

    # Validate that results are not wildly off
    assert np.allclose(acq_fast.mean(), 1.0, rtol=0.01)
    assert np.allclose(inf_fast.mean(), pars.r0 / 10.0, rtol=0.01)

    # Optional: alert if speedup is below expectation
    if speedup < 4.0:
        pytest.warns(UserWarning, f"Speedup lower than expected: {speedup:.2f}x")


#@pytest.mark.slow
def test_large_population_no_crash():
    N = 100_000_000
    acq = np.empty(N, dtype=np.float32)
    inf = np.empty(N, dtype=np.float32)
    pars = _make_pars(seed=123)

    populate_heterogeneous_values(0, N, acq, inf, pars)

    assert np.isfinite(acq).all()
    assert np.isfinite(inf).all()
