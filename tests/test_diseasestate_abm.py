from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from laser_core.propertyset import PropertySet

import laser_polio as lp
from laser_polio.run_sim import run_sim

test_dir = Path(__file__).parent
data_path = test_dir / "data"


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
            "stop_if_no_cases": False,  # Stop simulation if no cases are present
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
    dur_exp = 1
    dur_inf = 1
    pars = {
        "start_date": lp.date("2020-01-01"),
        "dur": 1,
        "n_ppl": np.array([1000, 500]),  # Two nodes with populations
        "cbr": np.array([30, 25]),  # Birth rate per 1000/year
        "r0_scalars": np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
        "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
        "init_immun": 0.0,  # 20% initially immune
        "init_prev": 0.0,  # 5% initially infected
        "dur_exp": lp.constant(value=dur_exp),  # Duration of the exposed state
        "dur_inf": lp.constant(value=dur_inf),  # Duration of the infectious state
        "p_paralysis": 1 / 2000,  # 1% paralysis probability
        "stop_if_no_cases": False,  # Stop simulation if no cases are present
    }
    pars_1day = PropertySet(pars)
    pars["dur"] = 2
    pars_2days = PropertySet(pars)
    sim_1day = lp.SEIR_ABM(pars_1day)
    sim_2days = lp.SEIR_ABM(pars_2days)
    sim_1day.components = [lp.DiseaseState_ABM]
    sim_2days.components = [lp.DiseaseState_ABM]

    # Test the initalized counts
    n_ppl = pars["n_ppl"].sum()
    assert np.all(sim_1day.people.exposure_timer[:n_ppl] == dur_exp), "The exposure timers should equal dur_exp after initialization"
    assert np.all(sim_1day.people.infection_timer[:n_ppl] == dur_inf), "The infection timers should equal dur_inf after initialization"

    # Test a sim with one day
    sim_1day.people.disease_state[:n_ppl] = 1  # Set all to Exposed
    sim_1day.run()  # Run for one day
    # Remember that results are not tallied in the DiseaseState_ABM component (results are tallied in Transmission_ABM) so we have to sum them up manually
    assert np.sum(sim_1day.people.disease_state == 0) == 0  # No one should be Susceptible
    assert np.sum(sim_1day.people.disease_state == 1) == 0  # No one should be Exposed
    assert np.sum(sim_1day.people.disease_state == 2) == n_ppl  # Everyone should be Infected
    assert np.sum(sim_1day.people.disease_state == 3) == 0  # No one should be Recovered

    # Test a sim with two days
    sim_2days.people.disease_state[:n_ppl] = 1  # Set all to Exposed
    sim_2days.run()  # Run for two days
    assert np.sum(sim_2days.people.disease_state == 0) == 0  # No one should be Susceptible
    assert np.sum(sim_2days.people.disease_state == 1) == 0  # No one should be Exposed
    assert np.sum(sim_2days.people.disease_state == 2) == 0  # No one should be Infected
    assert np.sum(sim_2days.people.disease_state == 3) == n_ppl  # Everyone should be Recovered


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
            "seasonal_amplitude": 0.125,  # Seasonal variation in transmission
            "seasonal_peak_doy": 180,  # Phase of seasonal variation
            "distances": np.array([[0, 1], [1, 0]]),  # Distance in km between nodes
            "gravity_k": 0.5,  # Gravity scaling constant
            "gravity_a": 1,  # Origin population exponent
            "gravity_b": 1,  # Destination population exponent
            "gravity_c": 2.0,  # Distance exponent
            "max_migr_frac": 0.01,  # Fraction of population that migrates
            "stop_if_no_cases": False,  # Stop simulation if no cases are present
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM, lp.VitalDynamics_ABM]

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


# Test Disease Progression with transmission
def test_disease_timers_with_trans_explicit():
    # Setup sim with 2 people & 1 seeded infection
    dur_exp = 2
    dur_inf = 3
    t_to_paralysis = 10
    pars = PropertySet(
        {
            "start_date": lp.date("2018-01-01"),
            "dur": 30,
            "n_ppl": np.array([1, 1]),  # Two nodes with populations
            "cbr": np.array([0, 0]),  # Birth rate per 1000/year
            "r0_scalars": np.array([1.0, 1.0]),  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": 0.0,  # initially immune <
            "init_prev": 1,  # initially infected from any age
            "dur_exp": lp.constant(value=dur_exp),  # Duration of the exposed state
            "dur_inf": lp.constant(value=dur_inf),  # Duration of the infectious state
            "t_to_paralysis": lp.constant(value=t_to_paralysis),  # Time to paralysis
            "p_paralysis": 1 / 2000,  # 1% paralysis probability
            "r0": 999,  # Basic reproduction number
            "distances": np.array([[0, 1], [1, 0]]),  # Distance in km between nodes
            "stop_if_no_cases": False,  # Stop simulation if no cases are present
            "seed": 123,
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM, lp.VitalDynamics_ABM]

    # Test the initalized counts
    disease_state = sim.people.disease_state[: sim.people.count]  # Filter to the active individuals
    n_s_init = np.sum(disease_state == 0)
    n_e_init = np.sum(disease_state == 1)
    n_i_init = np.sum(disease_state == 2)
    n_r_init = np.sum(disease_state == 3)
    assert n_s_init == 1, "There should be 1 S"
    assert n_e_init == 0, "There should be no E during initialization"
    assert n_i_init == 1, "There should be 1 I"
    assert n_r_init == 0, "There should be no R during initialization"

    # Test the timers
    e_timers = sim.people.exposure_timer[:]
    i_timers = sim.people.infection_timer[:]
    p_timers = sim.people.paralysis_timer[:]
    assert np.all(e_timers == dur_exp), "Exposure timers should be equal to dur_exp"
    assert np.all(i_timers == dur_inf), "Infection timers should be equal to dur_inf"
    assert np.all(p_timers == t_to_paralysis), "Paralysis timers should be equal to t_to_paralysis"

    # Run the simulation for one timestep
    sim.run()

    # Extract results
    n_s = np.sum(sim.results.S, axis=1)
    n_e = np.sum(sim.results.E, axis=1)
    n_i = np.sum(sim.results.I, axis=1)
    n_r = np.sum(sim.results.R, axis=1)
    n_p = np.sum(sim.results.new_potentially_paralyzed, axis=1)

    # Calc time to state changes
    zeros = np.zeros(sim.pars.dur + 1).astype(int)
    # S
    n_s_exp = zeros.copy()
    n_s_exp[0] = 1  # The susceptible (non-seeded infection) should start as S then become E on day 1 (super high r0)
    # E
    n_e_exp = zeros.copy()
    n_e_exp[1 : (1 + dur_exp)] = (
        1  # The susceptible (non-seeded infection) should become E on day 1 (super high r0) then recover after dur_exp days
    )
    # I
    n_i_exp = zeros.copy()
    n_i_exp[0 : (dur_inf + 1)] += 1  # The seeded infection starts as I and recovers after dur_inf days (+1 for day 0)
    n_i_exp[(1 + dur_exp) : (1 + dur_exp + dur_inf)] += (
        1  # new infection should become E on day 1 (super high r0) then recover after dur_inf days
    )
    # R
    n_r_exp = zeros.copy()
    n_r_exp[(1 + dur_inf) :] += 1  # seeded infection should recover after dur_inf days (+1 for day 0)
    n_r_exp[(1 + dur_exp + dur_inf) :] += 1  # new infections should recover after dur_exp days
    # P
    n_p_exp = zeros.copy()
    n_p_exp[1 + t_to_paralysis] += 1  # The seeded infection should become P after t_to_paralysis days (+1 for day 0)
    n_p_exp[2 + t_to_paralysis] += 1  # The new infection should become E on day 1 (super high r0) then become P after t_to_paralysis days

    # Check that the results match the expected counts
    assert np.all(n_s == n_s_exp), "S counts should match expected counts"
    assert np.all(n_e == n_e_exp), "E counts should match expected counts"
    assert np.all(n_i == n_i_exp), "I counts should match expected counts"
    assert np.all(n_r == n_r_exp), "R counts should match expected counts"
    assert np.all(n_p == n_p_exp), "P counts should match expected counts"


# Test Paralysis Probability
def test_paralysis_probability():
    """Ensure the correct fraction of infected individuals become paralyzed."""
    # Setup sim with 0 infections
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),
            "dur": 100,
            "n_ppl": np.array([50000, 50000]),  # Two nodes
            "cbr": np.array([30, 25]),  # Birth rate per 1000/year
            "r0_scalars": np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": 0.0,  # 20% initially immune
            "init_prev": 1.0,  # 5% initially infected
            "dur_exp": lp.constant(value=1),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "p_paralysis": 1 / 2000,  # 1% paralysis probability
            "stop_if_no_cases": False,  # Stop simulation if no cases are present
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM]
    sim.people.disease_state[: sim.pars.n_ppl.sum()] = 1  # Set all to Exposed
    sim.run()
    exp_paralysis = int(pars.p_paralysis * pars.n_ppl.sum())
    assert 0 < np.sum(sim.people.paralyzed == 1) <= exp_paralysis * 2  # Added some leeway for randomness


@patch("laser_polio.root", Path("tests/"))
def test_run_sim():
    sim = lp.run_sim(pop_scale=1 / 1000, n_days=3, verbose=0)
    assert isinstance(sim, lp.SEIR_ABM), "The simulation should be an instance of SEIR_ABM"
    assert hasattr(sim, "results"), "The simulation should have results"
    assert hasattr(sim, "pars"), "The simulation should have parameters"
    assert hasattr(sim, "people"), "The simulation should have people"
    assert hasattr(sim, "components"), "The simulation should have components"
    assert hasattr(sim, "run"), "The simulation should have a run method"
    assert hasattr(sim, "plot"), "The simulation should have a plot method"
    assert sim.results.S.shape[0] == sim.pars.dur + 1, "The results should have the same number of timesteps as the duration"
    assert sim.results.S.shape[1] == len(sim.pars.n_ppl), "The results should have the same number of nodes as the population"
    assert sim.results.I[0, 0] > 0  # Check that the initial infected count is greater than 0


@patch("laser_polio.root", Path("tests/"))
def test_seed_schedule():
    """Test infection seeding using dot_name + date via run_sim()."""

    regions = ["ZAMFARA"]
    start_year = 2019
    n_days = 30
    pop_scale = 1 / 100
    init_region = "ANKA"
    init_prev = 0.01
    r0 = 0  # Prevent transmission
    seed_schedule = [
        {"date": "2019-01-02", "dot_name": "AFRO:NIGERIA:ZAMFARA:BAKURA", "prevalence": 0.1},  # day 1
        {"date": "2019-01-03", "dot_name": "AFRO:NIGERIA:ZAMFARA:GUMMI", "prevalence": 0.1},  # day 2
        {"timestep": 3, "node_id": 13, "prevalence": 0.05},  # day 3
    ]

    sim = lp.run_sim(
        regions=regions,
        start_year=start_year,
        n_days=n_days,
        pop_scale=pop_scale,
        init_region=init_region,
        init_prev=init_prev,
        results_path=None,
        save_plots=False,
        save_data=False,
        verbose=0,
        r0=r0,
        vx_prob_ri=None,  # No vaccination
        vx_prob_sia=None,  # No vaccination
        seed_schedule=seed_schedule,
        age_pyramid_path=str(data_path / "Nigeria_age_pyramid_2024.csv"),
        stop_if_no_cases=False,  # Prevent early stopping for testing
    )

    # Test that all nodes except AKNA have 0 infections on day 0
    assert np.sum(sim.results.I[0, 1:]) == 0, "There should be no infections on day 0"

    # Check infections seeded on day 1 in BAKURA (node 1)
    dot_name = seed_schedule[0]["dot_name"]
    node_id = 1  # next((nid for nid, info in sim.pars.node_lookup.items() if info["dot_name"] == dot_name), None)
    t = 1
    n_inf_seed_schedule1 = sim.results.I[t, node_id]
    assert n_inf_seed_schedule1 > 0, f"No infections seeded with seed_schedule in {dot_name} at t={t}"

    # Check infections seeded on day 2 in GUMMI (node 5)
    dot_name = seed_schedule[1]["dot_name"]
    node_id = 5  # next((nid for nid, info in sim.pars.node_lookup.items() if info["dot_name"] == dot_name), None)
    t = 2
    n_inf_seed_schedule2 = sim.results.I[t, node_id]
    assert n_inf_seed_schedule2 > 0, f"No infections seeded with seed_schedule in {dot_name} at t={t}"

    # Check infections seeded on day 3 in node 13 ()
    dot_name = sim.pars.node_lookup[13]["dot_name"]
    node_id = 13  # next((nid for nid, info in sim.pars.node_lookup.items() if info["dot_name"] == dot_name), None)
    t = 3
    n_inf_seed_schedule3 = sim.results.I[t, node_id]
    assert n_inf_seed_schedule3 > 0, f"No infections seeded with seed_schedule in {dot_name} at t={t}"


@patch("laser_polio.root", Path("tests/"))
def test_init_immun_scalar():
    scalar = 0.5
    common_config = {
        "regions": ["ZAMFARA"],
        "start_year": 2018,
        "n_days": 1,
        "init_region": "ZAMFARA",
        "init_prev": 0,
        "pop_scale": 0.01,
        "results_path": None,
    }

    # Run unscaled version
    sim_unscaled = run_sim(config=common_config.copy(), run=False)
    df_unscaled = sim_unscaled.pars.init_immun.copy()
    cols = [col for col in df_unscaled.columns if col.startswith("immunity_")]

    # Run scaled version
    scaled_config = common_config.copy()
    scaled_config["init_immun_scalar"] = scalar
    sim_scaled = run_sim(config=scaled_config, run=False)
    df_scaled = sim_scaled.pars.init_immun.copy()

    # Check scaling: scaled = clip(unscaled * scalar)
    expected_scaled = df_unscaled[cols] * scalar
    expected_scaled = expected_scaled.clip(0.0, 1.0)

    pd.testing.assert_frame_equal(df_scaled[cols], expected_scaled, atol=1e-6, check_dtype=False)


def test_time_to_paralysis():
    sim = setup_sim()
    dist = sim.pars.t_to_paralysis(1000)
    p_timer = sim.people.paralysis_timer
    assert np.isclose(dist.mean(), 12.5, atol=3)
    assert np.isclose(dist.std(), 3.5, atol=3)
    assert np.isclose(p_timer.mean(), 12.5, atol=3)
    assert np.isclose(p_timer.std(), 3.5, atol=3)


def test_paralysis_progression_manual():
    # Setup sim with 0 infections
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),
            "dur": 3,
            "n_ppl": np.array([4, 4]),
            "cbr": np.array([0]),  # Birth rate per 1000/year
            "r0_scalars": np.array([0.0]),  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": 0.0,  # 20% initially immune
            "init_prev": 0.0,  # 5% initially infected
            "dur_exp": lp.constant(value=1),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "p_paralysis": 1.0,  # 1% paralysis probability
            "stop_if_no_cases": False,  # Stop simulation if no cases are present
        }
    )

    # Paralysis timer should trigger paralysis when not ipv_protected
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM]
    sim.people.disease_state[:] = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # Vary disease state
    sim.people.paralysis_timer[:] = 1  # Set paralysis timer to 1
    protected_idx = np.array([1, 3, 5, 7])
    unprotected_idx = np.setdiff1d(np.arange(sim.people.count), protected_idx)
    sim.people.ipv_protected[:] = 0
    sim.people.ipv_protected[protected_idx] = 1
    sim.run()

    # Those not ipv_protected should be paralyzed
    assert np.sum(sim.people.potentially_paralyzed[protected_idx] <= 0) == 4, (
        "SEIR people who are protected should not be potentially paralyzed"
    )
    assert np.sum(sim.people.potentially_paralyzed[unprotected_idx] > 0) == 3, (
        "EIR (not S) People who are not protected should be potentially paralyzed"
    )

    assert np.sum(sim.people.paralyzed[protected_idx] <= 0) == 4, "SEIR people who are protected should not be paralyzed"
    assert np.sum(sim.people.paralyzed[unprotected_idx] > 0) == 3, "EIR (not S) People who are not protected should be paralyzed"

    assert np.sum(sim.results.potentially_paralyzed[-1]) == 3, "Should have 3 potentially paralyzed individuals"
    assert np.sum(sim.results.paralyzed[-1]) == 3, "Should have 3 paralyzed individuals"


def test_paralysis_fraction_sans_ipv():
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),
            "dur": 60,
            "n_ppl": np.array([500000, 500000]),
            "cbr": np.array([0]),  # Birth rate per 1000/year
            "r0_scalars": np.array([1.0]),  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": 0.0,  # 20% initially immune
            "init_prev": 1.0,  # 5% initially infected
            "dur_exp": lp.constant(value=1),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "p_paralysis": 0.0005,  # 0.05% paralysis probability
            "stop_if_no_cases": False,  # Stop simulation if no cases are present
        }
    )

    # Paralysis timer should trigger paralysis when not ipv_protected
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM]
    sim.run()

    potential_paralyzed = np.sum(sim.results.potentially_paralyzed[-1])
    paralyzed = np.sum(sim.results.paralyzed[-1])
    sum_new_potential = np.sum(sim.results.new_potentially_paralyzed)
    sum_new_paralyzed = np.sum(sim.results.new_paralyzed)
    assert sum_new_potential == potential_paralyzed, "Potential paralyzed should be the sum of new potentially paralyzed"
    assert sum_new_paralyzed == paralyzed, "Paralyzed should be the sum of new paralyzed"
    assert np.isclose(potential_paralyzed / sim.pars.n_ppl.sum(), 1.0, atol=0.05), "Should have 100% potentially paralyzed"
    assert np.isclose(paralyzed / sim.pars.n_ppl.sum(), 0.0005, atol=0.001), "Should have 1/2000 paralyzed"


def test_paralysis_fraction_with_manual_ipv():
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),
            "dur": 60,
            "n_ppl": np.array([500000, 500000]),
            "cbr": np.array([0]),  # Birth rate per 1000/year
            "r0_scalars": np.array([1.0]),  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": 0.0,  # 20% initially immune
            "init_prev": 1.0,  # 5% initially infected
            "dur_exp": lp.constant(value=1),  # Duration of the exposed state
            "dur_inf": lp.constant(value=1),  # Duration of the infectious state
            "p_paralysis": 0.0005,  # 0.05% paralysis probability
            "stop_if_no_cases": False,  # Stop simulation if no cases are present
        }
    )

    # Paralysis timer should trigger paralysis when not ipv_protected
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM]
    sim.people.ipv_protected[:] = 1
    sim.run()

    potential_paralyzed = np.sum(sim.results.potentially_paralyzed[-1])
    paralyzed = np.sum(sim.results.paralyzed[-1])
    sum_new_potential = np.sum(sim.results.new_potentially_paralyzed)
    sum_new_paralyzed = np.sum(sim.results.new_paralyzed)
    assert sum_new_potential == potential_paralyzed, "Potential paralyzed should be the sum of new potentially paralyzed"
    assert sum_new_paralyzed == paralyzed, "Paralyzed should be the sum of new paralyzed"
    assert potential_paralyzed == 0, "Should have 100% potentially paralyzed"
    assert paralyzed == 0, "Should have 1/2000 paralyzed"


@patch("laser_polio.root", Path("tests/"))
def test_potential_paralysis():
    """Test that potential paralysis less than or equal to new exposed (due to IPV & delays in paralysis)."""

    regions = ["ZAMFARA"]
    start_year = 2018
    n_days = 365
    pop_scale = 1 / 1
    init_region = "ANKA"
    init_prev = 200
    r0 = 14
    migration_method = "radiation"
    radiation_k = 0.5
    max_migr_frac = 1.0
    verbose = 0
    vx_prob_ri = 0.0  # Sets OPV RI to 0, but allows IPV RI to be applied
    missed_frac = 0.1
    seed_schedule = [
        {"date": "2018-01-02", "dot_name": "AFRO:NIGERIA:ZAMFARA:BAKURA", "prevalence": 200},  # day 1
        {"date": "2018-11-07", "dot_name": "AFRO:NIGERIA:ZAMFARA:GUMMI", "prevalence": 200},  # day 2
    ]
    save_plots = False
    save_data = False
    plot_pars = False
    seed = 1
    # Diffs from demo_zamfara_load_init_pop.py
    results_path = "results/demo_zamfara"
    init_pop_file = None
    use_pim_scalars = True

    sim = lp.run_sim(
        regions=regions,
        start_year=start_year,
        n_days=n_days,
        pop_scale=pop_scale,
        init_region=init_region,
        init_prev=init_prev,
        results_path=results_path,
        save_plots=save_plots,
        save_data=save_data,
        plot_pars=plot_pars,
        verbose=verbose,
        seed=seed,
        r0=r0,
        migration_method=migration_method,
        radiation_k=radiation_k,
        max_migr_frac=max_migr_frac,
        vx_prob_ri=vx_prob_ri,
        init_pop_file=init_pop_file,
        seed_schedule=seed_schedule,
        missed_frac=missed_frac,
        use_pim_scalars=use_pim_scalars,
    )

    assert np.sum(sim.results.new_potentially_paralyzed) <= np.sum(sim.results.new_exposed), (
        "Potential paralysis should be less than or equal to new exposed"
    )
    assert np.isclose(np.sum(sim.results.new_potentially_paralyzed) / 2000, np.sum(sim.results.new_paralyzed), atol=17), (
        "Potential paralysis should be 1/2000 of new exposed"
    )


if __name__ == "__main__":
    test_disease_state_initialization()
    test_initial_population_counts()
    test_progression_without_transmission()
    test_progression_with_transmission()
    test_disease_timers_with_trans_explicit()
    test_paralysis_probability()
    test_run_sim()
    test_seed_schedule()
    test_init_immun_scalar()
    test_time_to_paralysis()
    test_paralysis_progression_manual()
    test_paralysis_fraction_sans_ipv()
    test_paralysis_fraction_with_manual_ipv()
    test_potential_paralysis()
    print("All disease state tests passed!")
