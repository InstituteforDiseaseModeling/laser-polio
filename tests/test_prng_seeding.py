from pathlib import Path
from unittest.mock import patch

import numpy as np

import laser_polio as lp

test_dir = Path(__file__).parent
data_path = test_dir / "data"


@patch("laser_polio.root", Path("tests/"))
def test_same_seed():
    """Test that the same seed produces the same results."""
    sim1 = lp.run_sim(seed=42)

    # Setup the second simulation with the same seed _after_ sim1.run()
    # otherwise sim1.run() will invalidate setting the global seed.
    sim2 = lp.run_sim(seed=42)

    assert np.all(sim1.results.births == sim2.results.births), "The two simulations should be identical (births), but they are not."
    assert np.all(sim1.results.deaths == sim2.results.deaths), "The two simulations should be identical (deaths), but they are not."
    assert np.all(sim1.results.paralyzed == sim2.results.paralyzed), (
        "The two simulations should be identical (paralyzed), but they are not."
    )
    assert np.all(sim1.results.ri_vaccinated == sim2.results.ri_vaccinated), (
        "The two simulations should be identical (ri_vaccinated), but they are not."
    )
    assert np.all(sim1.results.sia_protected == sim2.results.sia_protected), (
        "The two simulations should be identical (sia_protected), but they are not."
    )
    assert np.all(sim1.results.sia_vaccinated == sim2.results.sia_vaccinated), (
        "The two simulations should be identical (sia_vaccinated), but they are not."
    )


@patch("laser_polio.root", Path("tests/"))
def test_different_seeds():
    """Test that different seeds produce different results."""
    sim1 = lp.run_sim(seed=42)

    # Setup the second simulation with a different seed _after_ sim1.run()
    # otherwise sim1.run() will invalidate setting the global seed.
    sim2 = lp.run_sim(seed=13)

    same = (
        np.all(sim1.results.births == sim2.results.births)
        and np.all(sim1.results.deaths == sim2.results.deaths)
        and np.all(sim1.results.paralyzed == sim2.results.paralyzed)
        and np.all(sim1.results.ri_vaccinated == sim2.results.ri_vaccinated)
        and np.all(sim1.results.sia_protected == sim2.results.sia_protected)
        and np.all(sim1.results.sia_vaccinated == sim2.results.sia_vaccinated)
    )
    assert not same, "The two simulations should be different, but they are not."


if __name__ == "__main__":
    test_same_seed()
    test_different_seeds()
    print("All PRNGtests passed.")
