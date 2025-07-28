from pathlib import Path
from unittest.mock import patch

import numpy as np
import sciris as sc

from laser_polio.run_sim import run_sim

test_dir = Path(__file__).parent
data_path = test_dir / "data"


@patch("laser_polio.root", Path("tests/"))
def test_background_seeding():
    n_days = 90
    seeding_freq = 15
    sim = run_sim(
        regions=["ZAMFARA"],
        start_year=2018,
        n_days=n_days,  # short test run
        r0=14,
        init_prev=0.0,
        background_seeding=True,
        background_seeding_node_frac=1.0,  # all nodes
        background_seeding_freq=seeding_freq,
        background_seeding_prev=1,  # noticeable
        use_pim_scalars=False,
        verbose=0,
        run=True,
        save_data=False,
        save_plots=False,
    )

    # Reconstruct the expected seeding from the model parameters
    seed_schedule = sim.pars.seed_schedule
    node_lookup = sim.pars.node_lookup
    dot_to_node = {v["dot_name"]: k for k, v in node_lookup.items()}
    date_to_timestep = {d: t for t, d in enumerate(sc.daterange(sim.pars.start_date, days=sim.pars.dur + 1))}

    # Count hits
    n_matches = 0
    for entry in seed_schedule:
        dot = entry["dot_name"]
        seed_date = entry["date"]
        node = dot_to_node[dot]
        t = date_to_timestep.get(seed_date, None)

        if t is not None and sim.results.I[t, node] > 0:
            n_matches += 1

    n_nodes = len(sim.pars.init_pop)
    exp_n_seeds = (n_nodes * n_days) // seeding_freq  # There should be seeding event for every node every 15 days
    assert np.isclose(exp_n_seeds, len(seed_schedule), atol=5), "The length of the seed schedule did not meet expectations"

    total_seeds = len(seed_schedule)
    print(f"Matched infected nodes/times: {n_matches} / {total_seeds}")

    assert n_matches > 0, "No infections matched seeded locations and times"
    assert n_matches / total_seeds > 0.5, "Too few infections matched seeded targets"

    total_infected = np.sum(sim.results.I)
    print(f"Total infected: {total_infected}")
    assert total_infected > 0, "Background seeding failed to generate infections"


if __name__ == "__main__":
    test_background_seeding()
    print("All background seeding tests passed!")
