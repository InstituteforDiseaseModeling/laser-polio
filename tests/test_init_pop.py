import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from laser_core.random import seed as laser_seed

from laser_polio.run_sim import run_sim

test_dir = Path(__file__).parent
data_path = test_dir / "data"


@patch("laser_polio.root", Path("tests/"))
def test_init_pop_loading(tmp_path):
    init_dir = Path("tests/data/initpop_testcase")
    init_file = init_dir / "init_pop.h5"

    config = {
        "regions": ["ZAMFARA"],
        "start_year": 2018,
        "n_days": 60,
        "init_region": "ANKA",
        "init_prev": 0.01,
        "pop_scale": 1.0,
        "r0": 14,
        "radiation_k": 0.5,
        "migration_method": "radiation",
        "max_migr_frac": 1.0,
        "vx_prob_ri": None,
        "vx_prob_sia": None,
        "save_plots": False,
        "save_data": False,
        "save_pop": True,
        "seed": 123,
    }

    if not init_file.exists():
        init_dir.mkdir(parents=True, exist_ok=True)
        run_sim(**config, results_path=init_dir, run=False)

    # Load-from-disk sim
    sim_loaded = run_sim(
        init_pop_file=init_file,
        results_path=tmp_path / "loaded_run",
        run=False,
        **config,
    )

    # Setup a fresh sim
    sim_fresh = run_sim(
        init_pop_file=None,
        results_path=tmp_path / "fresh_run",
        run=False,
        **config,
    )

    # 1. Verify population LaserFrame matches after initialization
    for prop in sim_loaded.people.__dict__:
        if isinstance(sim_loaded.people.__dict__[prop], np.ndarray):
            a = sim_loaded.people.__dict__[prop][: sim_loaded.people.count]
            b = sim_fresh.people.__dict__[prop][: sim_fresh.people.count]
            assert np.array_equal(a, b), f"Mismatch in LaserFrame property '{prop}'."

    # Run the simulations to completion
    laser_seed(123)
    sim_loaded.run()
    laser_seed(123)
    sim_fresh.run()

    # 2. Final state similarity
    final_I_loaded = np.sum(sim_loaded.results.I[-1])
    final_I_fresh = np.sum(sim_fresh.results.I[-1])
    print(f"Final infected counts: Loaded={final_I_loaded}, Fresh={final_I_fresh}")
    assert np.isclose(final_I_loaded, final_I_fresh, rtol=0.1), "Final infected counts diverge too much."

    final_R_loaded = np.sum(sim_loaded.results.R[-1])
    final_R_fresh = np.sum(sim_fresh.results.R[-1])
    print(f"Final recovered counts: Loaded={final_R_loaded}, Fresh={final_R_fresh}")
    assert np.isclose(final_R_loaded, final_R_fresh, rtol=0.1), "Final recovered counts diverge too much."


if __name__ == "__main__":
    tmp_dir = Path(tempfile.mkdtemp())
    test_init_pop_loading(tmp_dir)
    print("Loading initialized population tests passed.")
