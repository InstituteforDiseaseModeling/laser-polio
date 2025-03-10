import numpy as np
import pytest
import laser_polio as lp
from laser_core.propertyset import PropertySet

def setup_sim():
    """Initialize a test simulation with DiseaseState_ABM component."""
    pars = PropertySet(dict(
        start_date      = lp.date('2020-01-01'),
        dur       = 30,
        n_ppl           = np.array([1000, 500]),  # Two nodes with populations
        cbr             = np.array([30, 25]),  # Birth rate per 1000/year
        beta_spatial    = np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
        age_pyramid_path= 'data/Nigeria_age_pyramid_2024.csv',  # From https://www.populationpyramid.net/nigeria/2024/
        init_immun      = 0.2,  # 20% initially immune
        init_prev       = 0.05,  # 5% initially infected
        dur_exp         = lp.normal(mean=3, std=1),  # Duration of the exposed state
        dur_inf         = lp.gamma(shape=4.51, scale=5.32),  # Duration of the infectious state
        p_paralysis     = 1 / 2000,  # 1% paralysis probability
    ))
    sim = lp.SEIR_ABM(pars)
    sim.components = [ lp.DiseaseState_ABM ]
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
    assert (total_pop - exp_rec - exp_inf) <= np.sum(sim.people.disease_state == 0) <= (total_pop - exp_rec), 'Susceptible counts are incorrect'
    assert np.sum(sim.people.disease_state == 1) == exp_exp, 'Exposed counts are incorrect'
    assert np.sum(sim.people.disease_state == 2) == exp_inf, 'Infected counts are incorrect'
    assert (exp_rec - exp_inf) <= np.sum(sim.people.disease_state == 3) <= exp_rec, 'Recovered counts are incorrect'

# Test Disease Progression
def test_progression_manual_seeding():
    # Setup sim with 0 infections
    pars = PropertySet(dict(
        start_date      = lp.date('2020-01-01'),
        dur       = 1,
        n_ppl           = np.array([1000, 500]),  # Two nodes with populations
        cbr             = np.array([30, 25]),  # Birth rate per 1000/year
        beta_spatial    = np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
        age_pyramid_path= 'data/Nigeria_age_pyramid_2024.csv',  # From https://www.populationpyramid.net/nigeria/2024/
        init_immun      = 0.0,  # 20% initially immune
        init_prev       = 0.0,  # 5% initially infected
        dur_exp         = lp.constant(value=1),  # Duration of the exposed state
        dur_inf         = lp.constant(value=1),  # Duration of the infectious state
        p_paralysis     = 1 / 2000,  # 1% paralysis probability
    ))
    sim = lp.SEIR_ABM(pars)
    assert np.all(sim.people.exposure_timer[:pars['n_ppl'].sum()] == 1), 'The exposure timer was not initialized correctly'
    assert np.all(sim.people.infection_timer[:pars['n_ppl'].sum()] == 1), 'The infection timer was not initialized correctly'
    sim.people.disease_state[:sim.pars.n_ppl.sum()] = 1  # Set all to Exposed
    sim.run()
    assert np.sum(sim.people.disease_state == 0) == np.sum(sim.results.S, axis=1) == 0  # No one should be Susceptible
    assert np.sum(sim.people.disease_state == 1) == np.sum(sim.results.E, axis=1) == 0  # No one should be Exposed
    assert np.sum(sim.people.disease_state == 2) == np.sum(sim.results.I, axis=1) == pars['n_ppl'].sum()  # Everyone should be Infected
    assert np.sum(sim.people.disease_state == 3) == np.sum(sim.results.R, axis=1) == 0  # No one should be Recovered
    sim.run()  # Run for another day
    assert np.sum(sim.people.disease_state == 0) == np.sum(sim.results.S, axis=1) == 0  # No one should be Susceptible
    assert np.sum(sim.people.disease_state == 1) == np.sum(sim.results.E, axis=1) == 0  # No one should be Exposed
    assert np.sum(sim.people.disease_state == 2) == np.sum(sim.results.I, axis=1) == 0  # No one should be Infected
    assert np.sum(sim.people.disease_state == 3) == np.sum(sim.results.R, axis=1) == pars['n_ppl'].sum()  # Everyone should be Recovered


# Test Disease Progression
def test_progression_with_transmission():
    # Setup sim with 0 infections
    pars = PropertySet(dict(
        start_date      = lp.date('2020-01-01'),
        dur       = 1,
        n_ppl           = np.array([1000, 500]),  # Two nodes with populations
        cbr             = np.array([30, 25]),  # Birth rate per 1000/year
        beta_spatial    = np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
        age_pyramid_path= 'data/Nigeria_age_pyramid_2024.csv',  # From https://www.populationpyramid.net/nigeria/2024/
        init_immun      = 0.5,  # 20% initially immune
        init_prev       = 0.1,  # 5% initially infected
        dur_exp         = lp.constant(value=1),  # Duration of the exposed state
        dur_inf         = lp.constant(value=1),  # Duration of the infectious state
        p_paralysis     = 1 / 2000,  # 1% paralysis probability
        r0              = 999,  # Basic reproduction number
        risk_mult_var   = 4.0,  # Lognormal variance for the individual-level risk multiplier (risk of acquisition multiplier; mean = 1.0)
        corr_risk_inf   = 0.8,  # Correlation between individual risk multiplier and individual infectivity (daily infectivity, mean = 14/24)
        seasonal_factor = 0.125,  # Seasonal variation in transmission
        seasonal_phase  = 180,  # Phase of seasonal variation
        distances       = np.array([[0,1],[1,0]]), # Distance in km between nodes
        gravity_k       = 0.5,  # Gravity scaling constant
        gravity_a       = 1,  # Origin population exponent
        gravity_b       = 1,  # Destination population exponent
        gravity_c       = 2.0,  # Distance exponent
        max_migr_frac   = 0.01, # Fraction of population that migrates
    ))
    sim = lp.SEIR_ABM(pars)
    sim.components = [ lp.DiseaseState_ABM, lp.Transmission_ABM ]

    # Ensure that there's a mix of disease states
    n_sus_t0 = np.sum(sim.people.disease_state == 0)
    n_exp_t0 = np.sum(sim.people.disease_state == 1)
    n_inf_t0 = np.sum(sim.people.disease_state == 2)
    n_rec_t0 = np.sum(sim.people.disease_state == 3)
    assert n_sus_t0 > 0
    assert n_exp_t0 == 0
    assert n_inf_t0 > 0
    assert n_rec_t0 > 0

    inf_idx = np.where(sim.people.disease_state == 2)[0]
    sim.people.infection_timer[inf_idx]

    exp_inx = np.where(sim.people.disease_state == 1)[0]
    sim.people.exposure_timer[exp_inx]

    # Run the simulation for one timestep
    sim.run()
    n_sus_t1 = np.sum(sim.people.disease_state == 0)
    n_exp_t1 = np.sum(sim.people.disease_state == 1)
    n_inf_t1 = np.sum(sim.people.disease_state == 2)
    n_rec_t1 = np.sum(sim.people.disease_state == 3)

    s = sim.results.S
    e = sim.results.E
    i = sim.results.I
    r = sim.results.R
    assert n_sus_t1 == n_sus_t0
    assert n_exp_t1 > 0
    assert n_inf_t1 == 0
    assert n_rec_t1 == n_rec_t0 + n_inf_t0

    # sim.run()  # Run for another day


# Test Paralysis Probability
def test_paralysis_probability():
    """Ensure the correct fraction of infected individuals become paralyzed."""
    # Setup sim with 0 infections
    pars = PropertySet(dict(
        start_date      = lp.date('2020-01-01'),
        dur       = 1,
        n_ppl           = np.array([10000, 5000]),  # Two nodes with populations
        cbr             = np.array([30, 25]),  # Birth rate per 1000/year
        beta_spatial    = np.array([0.5, 2.0]),  # Spatial transmission scalar (multiplied by global rate)
        age_pyramid_path= 'data/Nigeria_age_pyramid_2024.csv',  # From https://www.populationpyramid.net/nigeria/2024/
        init_immun      = 0.0,  # 20% initially immune
        init_prev       = 1.0,  # 5% initially infected
        dur_exp         = lp.constant(value=1),  # Duration of the exposed state
        dur_inf         = lp.constant(value=1),  # Duration of the infectious state
        p_paralysis     = 1 / 2000,  # 1% paralysis probability
    ))
    sim = lp.SEIR_ABM(pars)
    sim.components = [ lp.DiseaseState_ABM ]
    sim.people.disease_state[:sim.pars.n_ppl.sum()] = 1  # Set all to Exposed
    sim.run()
    exp_paralysis = int(pars.p_paralysis * pars.n_ppl.sum())
    assert 0 > np.sum(sim.people.paralyzed == 1) <= exp_paralysis * 2  # Added some leeway for randomness


if __name__ == '__main__':
    # test_disease_state_initialization()
    # test_initial_population_counts()
    # test_progression_manual_seeding()
    test_progression_with_transmission()
    # test_paralysis_probability()
    print('All disease state tests passed!')
