import numpy as np
from laser_core.propertyset import PropertySet

import laser_polio as lp

# TODO: (ask AI)
# Test no transmission when r0 = 0
# Test double transmission when r0x2 vs r0
# Test with different r0_scalarss
# Test impact of differnt dur_inf


def setup_sim(dur=1, n_ppl=None, r0_scalars=None, r0=14, dur_exp=None, dur_inf=None, init_immun=0.8, init_prev=0.01):
    if n_ppl is None:
        n_ppl = [10000, 10000]
    if r0_scalars is None:
        r0_scalars = [0.5, 2.0]
    if dur_exp is None:
        dur_exp = lp.constant(value=2)
    if dur_inf is None:
        dur_inf = lp.constant(value=1)
    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),
            "dur": dur,
            "n_ppl": n_ppl,  # Two nodes with populations
            "cbr": np.array([30, 25]),  # Birth rate per 1000/year
            "r0_scalars": r0_scalars,  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": init_immun,  # initially immune
            "init_prev": init_prev,  # initially infected from any age
            "dur_exp": dur_exp,  # Duration of the exposed state
            "dur_inf": dur_inf,  # Duration of the infectious state
            "p_paralysis": 1 / 2000,  # 1% paralysis probability
            "r0": r0,  # Basic reproduction number
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
    return sim


# Test default transmission scenario
def test_trans_default():
    sim = setup_sim()
    sim.run()
    assert sim.results.E[1:].sum() > 0, "There should be some exposures after the simulation runs."

    # Check if the number of exposures matches the expected value
    R0 = sim.pars["r0"]
    D = np.mean(sim.pars["dur_inf"](100))  # mean duration of infectiousness
    I = sim.results.I[0]  # initial infected individuals
    S = sim.results.S[0]  # susceptible individuals at the start
    N = sim.people.count
    exp_E = np.sum((R0 / D) * I * (S / N))
    obs_E = sim.results.E[1:].sum()
    assert np.isclose(obs_E, exp_E, atol=100), "The number of exposures does not match the expected value."


# Test ZERO transmission scenarios
def test_zero_trans():
    # Test with r0 = 0
    sim_r0_zero = setup_sim(r0=0)
    sim_r0_zero.run()
    assert sim_r0_zero.results.E[1:].sum() == 0, "There should be NO exposures when r0 is 0."

    # Test with r0_scalars = 0
    sim_r0_scalars_zero = setup_sim(r0_scalars=[0, 0])
    sim_r0_scalars_zero.run()
    assert sim_r0_scalars_zero.results.E[1:].sum() == 0, "There should be NO exposures when r0_scalars is 0."

    # Test with init_prev = 0
    sim_init_prev_zero = setup_sim(init_prev=0.0)
    sim_init_prev_zero.run()
    assert sim_init_prev_zero.results.E[1:].sum() == 0, "There should be NO exposures when init_prev is 0."


# Test DOUBLE transmission scenarios
def test_double_trans():
    # Default scenario
    init_immun = 0.0
    r0 = 5
    r0_scalars = np.array([1.0, 1.0])
    init_prev = 0.01
    sim_default = setup_sim(init_immun=init_immun, r0=r0, r0_scalars=r0_scalars, init_prev=init_prev)
    sim_default.run()

    # Double r0
    sim_r0_2x = setup_sim(init_immun=init_immun, r0=r0 * 2, r0_scalars=r0_scalars, init_prev=init_prev)
    sim_r0_2x.run()

    # Double r0_scalars
    sim_r0_scalars_2x = setup_sim(init_immun=init_immun, r0=r0, r0_scalars=r0_scalars * 2, init_prev=init_prev)
    sim_r0_scalars_2x.run()

    # Double init_prev
    sim_init_prev_2x = setup_sim(init_immun=init_immun, r0=r0, r0_scalars=r0_scalars, init_prev=init_prev * 2)
    sim_init_prev_2x.run()

    # Compare results
    n_e_t1_default = sim_default.results.E[1:].sum()
    n_e_t1_r0_2x = sim_r0_2x.results.E[1:].sum()
    n_e_t1_r0_scalars_2x = sim_r0_scalars_2x.results.E[1:].sum()
    n_e_t1_init_prev_2x = sim_init_prev_2x.results.E[1:].sum()
    atol = n_e_t1_default * 0.8  # Allow for some tolerance in the comparison
    assert np.isclose(n_e_t1_default * 2, n_e_t1_r0_2x, atol=atol), "Doubling r0 should approximately double the number of exposures."
    assert np.isclose(n_e_t1_default * 2, n_e_t1_r0_scalars_2x, atol=atol), (
        "Doubling r0_scalars should approximately double the number of exposures."
    )
    assert np.isclose(n_e_t1_default * 2, n_e_t1_init_prev_2x, atol=atol), (
        "Doubling init_prev should approximately double the number of exposures."
    )


if __name__ == "__main__":
    test_trans_default()
    test_zero_trans()
    test_double_trans()
    print("All transmission tests passed!")
