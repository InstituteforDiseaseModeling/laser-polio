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
        n_ppl = np.array([10000, 10000])
    if r0_scalars is None:
        r0_scalars = [0.5, 2.0]
    # if dur_exp is None:
    #     dur_exp = lp.constant(value=2)
    # if dur_inf is None:
    #     dur_inf = lp.constant(value=1)
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
            # "dur_exp": dur_exp,  # Duration of the exposed state
            # "dur_inf": dur_inf,  # Duration of the infectious state
            "p_paralysis": 1 / 2000,  # 1% paralysis probability
            "r0": r0,  # Basic reproduction number
            "risk_mult_var": 4.0,  # Lognormal variance for the individual-level risk multiplier (risk of acquisition multiplier; mean = 1.0)
            "corr_risk_inf": 0.8,  # Correlation between individual risk multiplier and individual infectivity (daily infectivity, mean = 14/24)
            "seasonal_amplitude": 0.0,  # Seasonal variation in transmission
            "seasonal_peak_doy": 180,  # Phase of seasonal variation
            "distances": np.array([[0, 1], [1, 0]]),  # Distance in km between nodes
            "gravity_k": 0.5,  # Gravity scaling constant
            "gravity_a": 1,  # Origin population exponent
            "gravity_b": 1,  # Destination population exponent
            "gravity_c": 2.0,  # Distance exponent
            "max_migr_frac": 0.01,  # Fraction of population that migrates
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM, lp.VitalDynamics_ABM]
    return sim


# Test default transmission scenario
def test_trans_default(n_reps=10):
    exposures = []
    for _ in range(n_reps):
        sim = setup_sim()
        sim.run()
        exposures.append(sim.results.E[1:].sum())
    exposures = np.array(exposures)  # ← Fix
    mean_obs_E = np.mean(exposures)

    # sim = setup_sim()
    # sim.run()
    assert np.all(exposures > 0), "There should be some exposures after the simulation runs."

    # Check if the number of exposures matches the expected value
    R0 = sim.pars["r0"]
    D = np.mean(sim.pars["dur_inf"](1000))  # mean duration of infectiousness
    S = sim.results.S[0]  # susceptible individuals at the start
    E = sim.results.E[0]  # susceptible individuals at the start
    I = sim.results.I[0]  # initial infected individuals
    R = sim.results.R[0]  # initial recovered individuals
    N = S + E + I + R  # total population
    # exp_E = np.sum((R0 / D) * I * (S / N))

    beta = R0 / D
    r0_scalars = sim.pars["r0_scalars"]
    lambda_ = beta * np.array(r0_scalars) * (I)
    per_agent_rate = lambda_ / N
    p_inf = 1 - np.exp(-per_agent_rate)
    exp_E = np.sum(S * p_inf)

    # Compare to binomial CIs
    stderr = np.sqrt(S * p_inf * (1 - p_inf)).sum()
    assert abs(mean_obs_E - exp_E) < 2 * stderr, "The mean number of exposures is not within 2 standard errors of the expected values."


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
def test_double_trans(n_reps=10):
    init_immun = 0.0
    r0 = 5
    r0_scalars = np.array([1.0, 1.0])
    init_prev = 0.01

    def run_exposures(r0_val, r0_scalars_val, init_prev_val):
        exposures = []
        for _ in range(n_reps):
            sim = setup_sim(init_immun=init_immun, r0=r0_val, r0_scalars=r0_scalars_val, init_prev=init_prev_val)
            sim.run()
            exposures.append(sim.results.E[1:].sum())
        return np.array(exposures)

    # Collect replicate exposures
    E_default = run_exposures(r0, r0_scalars, init_prev)
    E_r0_2x = run_exposures(r0 * 2, r0_scalars, init_prev)
    E_r0_scalars_2x = run_exposures(r0, r0_scalars * 2, init_prev)
    E_init_prev_2x = run_exposures(r0, r0_scalars, init_prev * 2)

    # Means and ratios
    mean_E_default = E_default.mean()
    tol = 0.2  # Accept 20% deviation from 2x ratio

    def check_ratio(name, doubled, label):
        ratio = doubled.mean() / mean_E_default
        assert np.isclose(ratio, 2.0, rtol=tol), f"Doubling {label} should approximately double exposures (got ratio={ratio:.2f})."

    check_ratio("r0", E_r0_2x, "r0")
    check_ratio("r0_scalars", E_r0_scalars_2x, "r0_scalars")
    check_ratio("init_prev", E_init_prev_2x, "init_prev")


if __name__ == "__main__":
    test_trans_default()
    test_zero_trans()
    test_double_trans()
    print("All transmission tests passed!")
