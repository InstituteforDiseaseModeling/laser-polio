import numpy as np
from laser_core.propertyset import PropertySet

import laser_polio as lp


def setup_sim():
    """Initialize a test simulation with DiseaseState_ABM component."""
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),
            "dur": 30,
            "n_ppl": np.array([1000, 500]),  # Two nodes with populations
            "cbr": np.array([30, 25]),  # Birth rate per 1000/year
            "r0_scalars": np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": 0.2,  # 20% initially immune
            "init_prev": 0.05,  # 5% initially infected
            "dur_exp": lp.normal(mean=3, std=1),  # Duration of the exposed state
            "dur_inf": lp.gamma(shape=4.51, scale=5.32),  # Duration of the infectious state
            "p_paralysis": 1 / 2000,  # 1% paralysis probability
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM]
    return sim


# Test Initialization
def test_disease_state_initialization():
    """Ensure disease state properties are correctly initialized."""
    sim = setup_sim()
    assert hasattr(sim.people, "disease_state")
    assert hasattr(sim.people, "paralyzed")
    assert hasattr(sim.people, "exposure_timer")
    assert hasattr(sim.people, "infection_timer")


# Test Initial Population Counts
def test_initial_population_counts():
    """Ensure the correct fraction of individuals start in each state."""
    sim = setup_sim()
    total_pop = np.sum(sim.pars.n_ppl)
    exp_rec = int(sim.pars.init_immun * total_pop)  # Note that some of the recovered can become infected during intitialization
    exp_inf = int(sim.pars.init_prev * total_pop)
    exp_exp = 0
    assert (total_pop - exp_rec - exp_inf) <= np.sum(sim.people.disease_state == 0) <= (total_pop - exp_rec), (
        "Susceptible counts are incorrect"
    )
    assert np.sum(sim.people.disease_state == 1) == exp_exp, "Exposed counts are incorrect"
    assert np.sum(sim.people.disease_state == 2) == exp_inf, "Infected counts are incorrect"
    assert (exp_rec - exp_inf) <= np.sum(sim.people.disease_state == 3) <= exp_rec, "Recovered counts are incorrect"


# Test Disease Progression without transmission
def test_progression_without_transmission():
    # Setup sim with 0 infections
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),
            "dur": 1,
            "n_ppl": np.array([1000, 500]),  # Two nodes with populations
            "cbr": np.array([30, 25]),  # Birth rate per 1000/year
            "r0_scalars": np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": 0.0,  # 20% initially immune
            "init_prev": 0.0,  # 5% initially infected
            "dur_exp": lp.constant(value=1),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "p_paralysis": 1 / 2000,  # 1% paralysis probability
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM]
    assert np.all(sim.people.exposure_timer[: pars["n_ppl"].sum()] == 0), "The exposure timer was not initialized correctly"
    assert np.all(sim.people.infection_timer[: pars["n_ppl"].sum()] == 1), "The infection timer was not initialized correctly"
    sim.people.disease_state[: sim.pars.n_ppl.sum()] = 1  # Set all to Exposed
    sim.run()  # Run for one day
    # Remember that results are not tallied in the DiseaseState_ABM component (results are tallied in Transmission_ABM) so we have to sum them up manually
    assert np.sum(sim.people.disease_state == 0) == 0  # No one should be Susceptible
    assert np.sum(sim.people.disease_state == 1) == 0  # No one should be Exposed
    assert np.sum(sim.people.disease_state == 2) == pars["n_ppl"].sum()  # Everyone should be Infected
    assert np.sum(sim.people.disease_state == 3) == 0  # No one should be Recovered
    sim.run()  # Run for another day
    assert np.sum(sim.people.disease_state == 0) == 0  # No one should be Susceptible
    assert np.sum(sim.people.disease_state == 1) == 0  # No one should be Exposed
    assert np.sum(sim.people.disease_state == 2) == 0  # No one should be Infected
    assert np.sum(sim.people.disease_state == 3) == pars["n_ppl"].sum()  # Everyone should be Recovered


# Test Disease Progression with transmission
def test_progression_with_transmission():
    # Setup sim with 0 infections
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),
            "dur": 6,
            "n_ppl": np.array([100, 50]),  # Two nodes with populations
            "cbr": np.array([30, 25]),  # Birth rate per 1000/year
            "r0_scalars": np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": 0.8,  # initially immune <
            "init_prev": 0.1,  # initially infected from any age
            "dur_exp": lp.constant(value=2),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "p_paralysis": 1 / 2000,  # 1% paralysis probability
            "r0": 999,  # Basic reproduction number
            "risk_mult_var": 4.0,  # Lognormal variance for the individual-level risk multiplier (risk of acquisition multiplier; mean = 1.0)
            "corr_risk_inf": 0.8,  # Correlation between individual risk multiplier and individual infectivity (daily infectivity, mean = 14/24)
            "seasonal_factor": 0.125,  # Seasonal variation in transmission
            "seasonal_phase": 180,  # Phase of seasonal variation
            "distances": np.array([[0, 1], [1, 0]]),  # Distance in km between nodes
            "gravity_k": 0.5,  # Gravity scaling constant
            "gravity_a": 1,  # Origin population exponent
            "gravity_b": 1,  # Destination population exponent
            "gravity_c": 2.0,  # Distance exponent
            "max_migr_frac": 0.01,  # Fraction of population that migrates
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM]

    # Test the initalized counts
    disease_state = sim.people.disease_state[: sim.people.count]  # Filter to the active individuals
    n_s_init = np.sum(disease_state == 0)
    n_e_init = np.sum(disease_state == 1)
    n_i_init = np.sum(disease_state == 2)
    n_r_init = np.sum(disease_state == 3)
    assert np.isclose(n_s_init, sim.pars.n_ppl.sum() * (1 - sim.pars.init_immun) * (1 - sim.pars.init_prev), atol=10), (
        "Roughly 18% of the pop should be S"
    )
    assert n_e_init == 0, "There should be no E during initialization"
    assert np.isclose(n_i_init, sim.pars.n_ppl.sum() * (sim.pars.init_prev), atol=5), "Roughly 10% of the population should be infected"
    assert np.isclose(n_r_init, sim.pars.n_ppl.sum() * (sim.pars.init_immun) * (1 - sim.pars.init_prev), atol=5), (
        "Roughly 72% of the pop should be R (since infections can override immunity in initialization)"
    )

    # Run the simulation for one timestep
    sim.run()

    # Check that the initialized counts match what's recorded on day 0 (we don't run step() on day 0, we only log results)
    n_s_t0 = sim.results.S[0].sum()
    n_e_t0 = sim.results.E[0].sum()
    n_i_t0 = sim.results.I[0].sum()
    n_r_t0 = sim.results.R[0].sum()
    assert n_s_t0 == n_s_init, "Counts on day 0 should be the same as during initialization"
    assert n_e_t0 == n_e_init, "Counts on day 0 should be the same as during initialization"
    assert n_i_t0 == n_i_init, "Counts on day 0 should be the same as during initialization"
    assert n_r_t0 == n_r_init, "Counts on day 0 should be the same as during initialization"

    # Check disease states at the end of day 1
    n_s_t1 = sim.results.S[1].sum()
    n_e_t1 = sim.results.E[1].sum()
    n_i_t1 = sim.results.I[1].sum()
    n_r_t1 = sim.results.R[1].sum()
    assert n_s_t1 < n_s_t0, "Some should have become exposed"
    assert n_e_t1 > 0, "Some should have become exposed since transmission occurred at the end of day 1"
    assert n_i_t1 == n_i_t0, "Infected counts should remain the same since dur_inf = 1"
    assert n_r_t1 == n_r_t0, "Recovered counts should remain the same"

    # Check disease states at the end of day 2
    n_s_t2 = sim.results.S[2].sum()
    n_e_t2 = sim.results.E[2].sum()
    n_i_t2 = sim.results.I[2].sum()
    n_r_t2 = sim.results.R[2].sum()
    assert n_s_t2 == n_s_t1, "Susceptible counts should be the same since no one should be infected"
    assert n_e_t2 == n_e_t1, "Exposed counts should be the same since dur_exp = 2 & no one should be infected to make more E"
    assert n_i_t2 == 0, "Infected counts should be 0 since dur_inf = 1 & exposed won't become infected until the next timestep"
    assert n_r_t2 == n_i_t1 + n_r_t1, "Recovered counts should be higher since I's should've recovered"

    # Check disease states at the end of day 3
    n_s_t3 = sim.results.S[3].sum()
    n_e_t3 = sim.results.E[3].sum()
    n_i_t3 = sim.results.I[3].sum()
    n_r_t3 = sim.results.R[3].sum()
    assert n_s_t3 == n_s_t2 - n_e_t3, "S's should be the same as yesterday minus those who became exposed"
    assert n_i_t3 == n_e_t2, "I's should be equal to the number of E from yesterday"
    assert n_r_t3 == n_r_t2, "No one should be recovered since there were no infected on the previous day"

    # Check disease states at the end of day 4
    n_s_t4 = sim.results.S[4].sum()
    n_e_t4 = sim.results.E[4].sum()
    n_i_t4 = sim.results.I[4].sum()
    n_r_t4 = sim.results.R[4].sum()
    assert n_s_t4 == n_s_t3, "S's should be the same as yesterday since there shouldn't be any I to make new E"
    assert n_e_t4 == n_e_t3, "E's should be the same as yesterday since there shouldn't be any I to make new E"
    assert n_i_t4 == 0, "I's should be 0 since dur_inf = 1 & E won't become E until the next timestep"
    assert n_r_t4 == n_r_t3 + n_i_t3, "R counts should be higher since I's should've become R"

    # Check disease states at the end of day 5
    n_s_t5 = sim.results.S[5].sum()
    n_e_t5 = sim.results.E[5].sum()
    n_i_t5 = sim.results.I[5].sum()
    n_r_t5 = sim.results.R[5].sum()
    assert n_s_t5 == n_s_t4 - n_e_t5, "S's should be the same as yesterday minus those who became exposed"
    assert n_i_t5 == n_e_t4, "I's should be equal to the number of E from yesterday"
    assert n_r_t5 == n_r_t4, "No one should be recovered since there were no infected on the previous day"

    # Check disease states at the end of day 6
    n_s_t6 = sim.results.S[6].sum()
    n_e_t6 = sim.results.E[6].sum()
    n_i_t6 = sim.results.I[6].sum()
    n_r_t6 = sim.results.R[6].sum()
    assert n_s_t6 == n_s_t5, "S's should be the same as yesterday since there shouldn't be any I to make new E"
    assert n_e_t6 == n_e_t5, "E's should be the same as yesterday since there shouldn't be any I to make new E"
    assert n_i_t6 == 0, "I's should be 0 since dur_inf = 1 & E won't become E until the next timestep"
    assert n_r_t6 == n_r_t5 + n_i_t5, "R counts should be higher since I's should've become R"

    # Check that the end state counts match what's recorded on day 6
    disease_state = sim.people.disease_state[: sim.people.count]
    n_s_end = np.sum(disease_state == 0)
    n_e_end = np.sum(disease_state == 1)
    n_i_end = np.sum(disease_state == 2)
    n_r_end = np.sum(disease_state == 3)
    assert n_s_end == n_s_t6, "Day 6 state counts should match logged results"
    assert n_e_end == n_e_t6, "Day 6 state counts should match logged results"
    assert n_i_end == n_i_t6, "Day 6 state counts should match logged results"
    assert n_r_end == n_r_t6, "Day 6 state counts should match logged results"


# Test Paralysis Probability
def test_paralysis_probability():
    """Ensure the correct fraction of infected individuals become paralyzed."""
    # Setup sim with 0 infections
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),
            "dur": 1,
            "n_ppl": np.array([50000, 50000]),  # Two nodes
            "cbr": np.array([30, 25]),  # Birth rate per 1000/year
            "r0_scalars": np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": 0.0,  # 20% initially immune
            "init_prev": 1.0,  # 5% initially infected
            "dur_exp": lp.constant(value=1),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "p_paralysis": 1 / 2000,  # 1% paralysis probability
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM]
    sim.people.disease_state[: sim.pars.n_ppl.sum()] = 1  # Set all to Exposed
    sim.run()
    exp_paralysis = int(pars.p_paralysis * pars.n_ppl.sum())
    assert 0 < np.sum(sim.people.paralyzed == 1) <= exp_paralysis * 2  # Added some leeway for randomness


if __name__ == "__main__":
    # test_disease_state_initialization()
    # test_initial_population_counts()
    # test_progression_without_transmission()
    # test_progression_with_transmission()
    test_paralysis_probability()
    print("All disease state tests passed!")
