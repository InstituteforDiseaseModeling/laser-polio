import tempfile
from pathlib import Path

import pandas as pd
import pytest

import laser_polio as lp
from laser_polio.run_sim import run_sim
from laser_polio.utils import save_sim_results


@pytest.mark.parametrize("ext", [".csv", ".h5"])
def test_save_sim_results(ext):
    # Run a tiny real simulation
    lp.root = Path()
    sim = run_sim(
        regions=["ZAMFARA"],
        start_year=2018,
        n_days=10,  # quick run
        r0=12,
        init_prev=0.01,
        background_seeding=False,
        use_pim_scalars=False,
        verbose=0,
        run=True,
        save_data=False,
        save_plots=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = Path(tmpdir) / f"test_output{ext}"

        # Save results
        df = save_sim_results(sim, outfile)

        # Basic checks
        assert outfile.exists(), f"File {outfile} was not created"
        assert not df.empty, "Saved DataFrame is empty"

        # Required columns — customize as needed
        expected_columns = {
            "timestep",
            "date",
            "node",
            "dot_name",
            "S",
            "E",
            "I",
            "R",
            "P",
            "births",
            "deaths",
            "new_exposed",
            "potentially_paralyzed",
            "new_potentially_paralyzed",
            "new_paralyzed",
        }
        assert expected_columns.issubset(df.columns), f"Missing columns: {expected_columns - set(df.columns)}"

        # Round-trip read
        if ext == ".csv":
            df2 = pd.read_csv(outfile, parse_dates=["date"])
        else:
            df2 = pd.read_hdf(outfile)

        # Equality checks
        assert len(df2) == len(df), f"Row count mismatch: {len(df2)} vs {len(df)}"
        assert set(df2.columns) == set(df.columns), "Column mismatch"

        # Spot check values — add more if needed
        assert df2["I"].sum() >= 0, "Infected count should be non-negative"
        assert df2["timestep"].max() < sim.nt, "Timestep values exceed simulation bounds"

        # Group-wise checks (e.g. ensure every timestep/node pair exists)
        expected_rows = sim.nt * len(sim.nodes)
        assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

        # Dot name uniqueness
        assert df["dot_name"].nunique() == len(sim.nodes), "Unexpected number of dot_name values"
