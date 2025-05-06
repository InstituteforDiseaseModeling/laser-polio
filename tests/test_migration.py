from pathlib import Path
from unittest.mock import patch

import numpy as np

from laser_polio.run_sim import run_sim

test_dir = Path(__file__).parent
data_path = test_dir / "data"


@patch("laser_polio.root", Path("tests/"))
def test_radiation(n_reps=5, duration=30, low_k=0.02, high_k=0.1, min_diff=2):
    """
    Run n_reps sims with 3 values of radiation_k: 0.0, low_k, and high_k.
    Confirm that spatial spread increases with radiation_k.
    """
    spread_zero = []
    spread_low = []
    spread_high = []

    for i in range(n_reps):
        seed = 1000 + i
        base_args = {
            "regions": ["ZAMFARA"],
            "start_year": 2018,
            "n_days": duration,
            "init_region": "ANKA",
            "init_prev": 0.01,
            "pop_scale": 0.1,
            "r0": 14,
            "migration_method": "radiation",
            "max_migr_frac": 1.0,
            "verbose": 0,
            "vx_prob_ri": None,
            "vx_prob_sia": None,
            "run": True,
            "save_plots": False,
            "save_data": False,
            "seed": seed,
        }

        # No migration
        sim_zero = run_sim(radiation_k=0.0, **base_args)
        infected_zero = np.sum(sim_zero.results.I, axis=0)
        spread_zero.append(np.count_nonzero(infected_zero > 0))

        # Low migration
        sim_low = run_sim(radiation_k=low_k, **base_args)
        infected_low = np.sum(sim_low.results.I, axis=0)
        spread_low.append(np.count_nonzero(infected_low > 0))

        # High migration
        sim_high = run_sim(radiation_k=high_k, **base_args)
        infected_high = np.sum(sim_high.results.I, axis=0)
        spread_high.append(np.count_nonzero(infected_high > 0))

    # Means
    mean_zero = np.mean(spread_zero)
    mean_low = np.mean(spread_low)
    mean_high = np.mean(spread_high)

    print(f"Mean nodes infected @ k=0.0   → {mean_zero:.2f}")
    print(f"Mean nodes infected @ k={low_k} → {mean_low:.2f}")
    print(f"Mean nodes infected @ k={high_k} → {mean_high:.2f}")

    # Assertions
    assert mean_zero == 1.0, f"Expected infection to stay in home node at k=0.0, but got mean {mean_zero:.2f} nodes infected."
    assert mean_low > mean_zero, f"Expected low_k to spread to more nodes than k=0.0, got {mean_low:.2f} vs {mean_zero:.2f}"
    assert mean_high >= mean_low + min_diff, (
        f"Expected high_k to infect at least {min_diff} more nodes than low_k, got {mean_high:.2f} vs {mean_low:.2f}"
    )


if __name__ == "__main__":
    test_radiation()
    print("All migration tests passed.")
