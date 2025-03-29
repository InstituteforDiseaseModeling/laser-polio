import numpy as np
from laser_core.propertyset import PropertySet

import laser_polio as lp


# @pytest.fixture
def setup_sim(step_size=1):
    """Initialize a small test simulation with birth & death tracking."""
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),  # Start date of the simulation
            "dur": 30,  # Number of dur to run the simulation
            "n_ppl": np.array([10000, 5000]),  # Two nodes with populations
            "cbr": np.array([30, 25]),  # Birth rate per 1000/year
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "step_size_VitalDynamics_ABM": step_size,
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM]
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
    # Tally births per day, add 1 since results include init conditions and dur timesteps
    expected_births = np.bincount(dobs, minlength=sim.pars.dur + 1)
    observed_births = np.sum(sim.results.births, axis=1)  # Sum across nodes per timestep
    # Combine births on day 0 and day 1 since we don't run vital rates on day 0, we only log results
    expected_births[1] += expected_births[0]  # Combine day 0 and day 1
    expected_births = expected_births[1:]  # Remove day 0 since it's combined with day 1
    observed_births[1] += observed_births[0]  # Combine day 0 and day 1
    observed_births = observed_births[1:]  # Remove day 0 since it's combined with day 1
    assert np.array_equal(expected_births, observed_births), "Births did not occur on the expected days"
    # print("Births:", sim.results.births)


# Test Deaths when timestep = 1
def test_deaths_occur_step_size_1():
    """Ensure deaths are correctly logged and marked."""
    sim = setup_sim(step_size=1)
    sim.people.date_of_death[:5] = 1  # Force first 5 agents to die at timestep 1, other deaths will be random
    sim.run()
    assert np.all(sim.people.date_of_death[:5] == 1), "First 5 dates of death were not set correctly"
    assert np.all(sim.people.disease_state[:5] == -1), "First 5 were not marked dead with disease_state"

    dods = sim.people.date_of_death[: sim.people.count]  # Extract date_of_death; only consider active individuals
    # Tally deaths per day, add 1 since results are include init conditions and dur timesteps
    expected_deaths = np.bincount(dods)[: sim.pars.dur + 1]
    observed_deaths = np.sum(sim.results.deaths, axis=1)  # Sum across nodes per timestep
    # Combine deaths on day 0 and day 1 since we don't run vital rates on day 0, we only log results
    expected_deaths[1] += expected_deaths[0]  # Combine day 0 and day 1
    expected_deaths = expected_deaths[1:]  # Remove day 0 since it's combined with day 1
    observed_deaths[1] += observed_deaths[0]  # Combine day 0 and day 1
    observed_deaths = observed_deaths[1:]  # Remove day 0 since it's combined with day 1
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
    bin_edges = np.arange(0, sim.pars.dur + 1, 7)  # Define bin edges
    dods = sim.people.date_of_death[: sim.people.count]  # Extract date_of_death for active individuals
    expected_deaths = np.bincount(dods, minlength=sim.pars.dur + 1)[: sim.pars.dur + 1]  # Daily expected deaths
    observed_deaths = np.sum(sim.results.deaths, axis=1)  # Daily observed deaths
    # Combine deaths on day 0 and day 1 since we don't run vital rates on day 0, we only log results
    expected_deaths[1] += expected_deaths[0]  # Combine day 0 and day 1
    expected_deaths[0] = 0  # Set to 0 since nothing run this day
    observed_deaths[1] += observed_deaths[0]  # Combine day 0 and day 1
    observed_deaths[0] = 0  # Set to 0 since nothing run this day
    death_bins = np.digitize(np.arange(sim.pars.dur + 1), bin_edges, right=True)  # Digitize days into bins
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
    assert np.all(end_ages == initial_ages + sim.pars.dur + 1)


# Test Edge Case: Zero Birth Rate
def test_zero_birth_rate():
    """Ensure that setting birth rate to zero prevents population growth."""
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),  # Start date of the simulation
            "dur": 30,  # Number of dur to run the simulation
            "n_ppl": np.array([1000, 500]),  # Two nodes with populations
            "cbr": np.array([0, 0]),  # Birth rate per 1000/year
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "step_size_VitalDynamics_ABM": 1,
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM]

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
