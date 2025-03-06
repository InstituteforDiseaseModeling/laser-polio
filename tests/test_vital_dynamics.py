import numpy as np
from laser_core.propertyset import PropertySet

import laser_polio as lp


# @pytest.fixture
def setup_sim(step_size=1):
    """Initialize a small test simulation with birth & death tracking."""
    pars = PropertySet(
        dict(  # noqa: C408
            start_date=lp.date("2020-01-01"),  # Start date of the simulation
            timesteps=30,  # Number of timesteps to run the simulation
            n_ppl=np.array([1000, 500]),  # Two nodes with populations
            cbr=np.array([30, 25]),  # Birth rate per 1000/year
            age_pyramid_path="data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
        )
    )
    sim = lp.SEIR_ABM(pars)
    sim.add_component(lp.VitalDynamics_ABM(sim, step_size=step_size))
    return sim


# Test Initialization
def test_vital_dynamics_initialization():
    """Ensure vital dynamics properties are correctly initialized."""
    sim = setup_sim()
    assert hasattr(sim.people, "date_of_birth")
    assert hasattr(sim.people, "date_of_death")
    assert hasattr(sim.results, "births")
    assert hasattr(sim.results, "deaths")


#  Test Births Over Time
def test_births_generated():
    """Verify that births occur at the expected rate."""
    step_size = 1
    sim = setup_sim(step_size=step_size)
    initial_population = sim.people.count
    sim.run()
    assert sim.people.count > initial_population, "Population did not increase despite high birth rates"
    assert initial_population + sim.results.births.sum() - sim.results.deaths.sum() == np.sum(sim.people.disease_state > -1), (
        "Population did not increase by the correct amount"
    )

    dobs = sim.people.date_of_birth[: sim.people.count]  # Extract date_of_birth; only consider active individuals
    dobs = dobs[dobs >= 0]  # Remove negatives (pop initialized before sim)
    expected_births = np.bincount(dobs, minlength=sim.pars.timesteps)  # Tally births per day
    observed_births = np.sum(sim.results.births, axis=1)  # Sum across nodes per timestep
    assert np.array_equal(expected_births, observed_births), "Births did not occur on the expected days"
    print("Births:", sim.results.births)


# Test Deaths when timestep = 1
def test_deaths_occur_step_size_1():
    """Ensure deaths are correctly logged and marked."""
    sim = setup_sim(step_size=1)
    sim.people.date_of_death[:5] = 1  # Force first 5 agents to die at timestep 1, other deaths will be random
    sim.run()
    assert np.all(sim.people.date_of_death[:5] == 1), "First 5 dates of death were not set correctly"
    assert np.all(sim.people.disease_state[:5] == -1), "First 5 were not marked dead with disease_state"

    dods = sim.people.date_of_death[: sim.people.count]  # Extract date_of_death; only consider active individuals
    expected_deaths = np.bincount(dods)[: sim.pars.timesteps]  # Tally deaths per day
    observed_deaths = np.sum(sim.results.deaths, axis=1)  # Sum across nodes per timestep
    assert np.array_equal(expected_deaths, observed_deaths), "Deaths did not occur on the expected days"

    unique_values, counts = np.unique(sim.people.disease_state[: np.sum(sim.pars.n_ppl)], return_counts=True)
    assert counts[0] == np.sum(sim.results.deaths), "Total number of dead agents does not match deaths logged"


# Test Deaths when timestep = 7
def test_deaths_occur_step_size_7():
    """Ensure deaths are correctly logged and marked with a timestep > 1."""
    step_size = 7
    sim = setup_sim(step_size=step_size)  # Step size of 7 days
    sim.people.date_of_death[:5] = 1  # Force first 5 agents to die at timestep 1, other deaths will be random
    sim.run()
    assert np.all(sim.people.date_of_death[:5] == 1), "First 5 dates of death were not set correctly"
    assert np.all(sim.people.disease_state[:5] == -1), "First 5 were not marked dead with disease_state"

    # Check that deaths occurred on the correct dates and were correctly aggregated into weekly bins
    bin_edges = np.arange(0, sim.pars.timesteps + 1, 7)  # Define bin edges
    dods = sim.people.date_of_death[: sim.people.count]  # Extract date_of_death for active individuals
    expected_deaths = np.bincount(dods, minlength=sim.pars.timesteps)[: sim.pars.timesteps]  # Daily expected deaths
    observed_deaths = np.sum(sim.results.deaths, axis=1)  # Daily observed deaths
    death_bins = np.digitize(np.arange(sim.pars.timesteps), bin_edges, right=True)  # Digitize days into bins
    expected_deaths_binned = np.bincount(death_bins, weights=expected_deaths)[: len(bin_edges) - 1]  # Aggregate deaths per bin
    observed_deaths_binned = np.bincount(death_bins, weights=observed_deaths)[: len(bin_edges) - 1]  # Aggregate deaths per bin
    assert np.array_equal(expected_deaths_binned, observed_deaths_binned), "Deaths did not occur on the expected days"

    unique_values, counts = np.unique(sim.people.disease_state[: np.sum(sim.pars.n_ppl)], return_counts=True)
    assert counts[0] == np.sum(sim.results.deaths), "Total number of dead agents does not match deaths logged"


# Test Age Progression
def test_age_progression():
    """Ensure age increases over time correctly."""
    sim = setup_sim()
    initial_ages = (
        sim.t - sim.people.date_of_birth[: np.sum(sim.pars.n_ppl)]
    )  # Only focus on active individuals. Unborn agents don't age until born
    sim.run()
    end_ages = sim.t - sim.people.date_of_birth[: np.sum(sim.pars.n_ppl)]
    assert np.all(end_ages == initial_ages + sim.pars.timesteps)


# Test Edge Case: Zero Birth Rate
def test_zero_birth_rate():
    """Ensure that setting birth rate to zero prevents population growth."""
    pars = PropertySet(
        dict(  # noqa: C408
            start_date=lp.date("2020-01-01"),  # Start date of the simulation
            timesteps=30,  # Number of timesteps to run the simulation
            n_ppl=np.array([1000, 500]),  # Two nodes with populations
            cbr=np.array([0]),  # Birth rate per 1000/year
            age_pyramid_path="data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
        )
    )
    sim = lp.SEIR_ABM(pars)
    sim.add_component(lp.VitalDynamics_ABM(sim, step_size=1))
    initial_population = sim.people.count
    sim.run()
    assert sim.people.count == initial_population  # No new births


if __name__ == "__main__":
    test_vital_dynamics_initialization()
    test_births_generated()
    test_deaths_occur_step_size_1()
    test_deaths_occur_step_size_7()
    test_age_progression()
    test_zero_birth_rate()
    print("All time tests passed.")
