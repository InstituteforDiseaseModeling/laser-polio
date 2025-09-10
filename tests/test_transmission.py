from pathlib import Path
from unittest.mock import patch

import numpy as np
from laser_core.propertyset import PropertySet
from scipy import stats
from scipy.stats import spearmanr

import laser_polio as lp

test_dir = Path(__file__).parent
data_path = test_dir / "data"

# TODO: (ask AI)
# Test no transmission when r0 = 0
# Test double transmission when r0x2 vs r0
# Test with different r0_scalarss
# Test impact of differnt dur_inf


def setup_sim(dur=1, init_pop=None, r0_scalars=None, r0=14, dur_exp=None, dur_inf=None, init_immun=0.8, init_prev=0.01, seed=None):
    if init_pop is None:
        init_pop = np.array([10000, 10000])
    if r0_scalars is None:
        r0_scalars = np.array([0.5, 2.0], dtype=np.float32)
    # if dur_exp is None:
    #     dur_exp = lp.constant(value=2)
    # if dur_inf is None:
    #     dur_inf = lp.constant(value=1)
    pars = PropertySet(
        {
            "seed": seed,
            "start_date": lp.date("2020-01-01"),
            "dur": dur,
            "init_pop": init_pop,  # Two nodes with populations
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
def test_double_trans():
    init_immun = 0.0
    r0 = 5
    r0_scalars = np.array([1.0, 1.0])
    init_prev = 0.01
    seeds = [12345, 23456, 34567, 45678, 56789, 67890, 78901, 89012, 90123, 11234]  # Fixed seeds for reproducibility

    def run_exposures(r0_val, r0_scalars_val, init_prev_val):
        exposures = []
        for seed in seeds:
            sim = setup_sim(init_immun=init_immun, r0=r0_val, r0_scalars=r0_scalars_val, init_prev=init_prev_val, seed=seed)
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


def setup_NxN_sim(N=4, duration=365, r0=14, init_immun=None, init_prev=None):
    if init_immun is None:
        init_immun = np.array([0.8] * N, dtype=np.float32)
    if init_prev is None:
        init_prev = np.array([0.01] * N, dtype=np.float32)

    pars = PropertySet(
        {
            "start_date": lp.date("2020-01-01"),
            "dur": duration,
            "init_pop": np.array([10000] * N),
            "cbr": np.array([36.5] * N, dtype=np.float32),  # Birth rate per 1000/year
            "r0_scalars": np.ones(N, dtype=np.float32),  # Spatial transmission scalar (multiplied by global rate)
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "init_immun": init_immun,  # initially immune
            "init_prev": init_prev,  # initially infected from any age
            "p_paralysis": 1 / 2000,  # 1% paralysis probability
            "r0": r0,  # Basic reproduction number
            "risk_mult_var": 4.0,  # Lognormal variance for the individual-level risk multiplier (risk of acquisition multiplier; mean = 1.0)
            "corr_risk_inf": 0.8,  # Correlation between individual risk multiplier and individual infectivity (daily infectivity, mean = 14/24)
            "seasonal_amplitude": 0.0,  # Seasonal variation in transmission
            "seasonal_peak_doy": 180,  # Phase of seasonal variation
            "distances": np.ones((N, N)) * 50,  # Distance in km between nodes
            "gravity_k": 0.5,  # Gravity scaling constant
            "gravity_a": 1,  # Origin population exponent
            "gravity_b": 1,  # Destination population exponent
            "gravity_c": 2.0,  # Distance exponent
            "max_migr_frac": 0.01,  # Fraction of population that migrates
            "dur_exp": lp.normal(mean=7, std=1),  # arbitrarily a bit longer than the default
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM, lp.VitalDynamics_ABM]

    return sim


def test_linear_transmission():
    # Initially only node 0 has infected
    sim = setup_NxN_sim(N=4, init_prev=np.array([0.05, 0.0, 0.0, 0.0], dtype=np.float32))

    # Modify network to transmit 0 -> 1 -> 2 -> 3
    index = next(i for i, inst in enumerate(sim.instances) if isinstance(inst, lp.Transmission_ABM))
    tx = sim.instances[index]
    tx.network = np.array(
        [
            [0, 1, 0, 0],  # Node 0 can only transmit to Node 1
            [0, 0, 1, 0],  # Node 1 can only transmit to Node 2
            [0, 0, 0, 1],  # Node 2 can only transmit to Node 3
            [0, 0, 0, 0],  # Node 3 cannot transmit to anyone
        ],
        dtype=tx.network.dtype,
    )

    sim.run()

    # Assert: Node 0 infections should decay to zero and then stay that way
    I_node0 = sim.results.I[:, 0]
    assert np.all(np.diff(I_node0) <= 0), "Infections in node 0 should not increase - all contagion is going to node 1."
    # Look for the first timestep with zero infections and ensure it remains zero
    first_zero_idx = np.where(I_node0 == 0)[0][0]
    assert np.all(I_node0[first_zero_idx:] == 0), "Node 0 infections should decay to zero and stay there."

    # Assert: Node 1 infections should peak ~25+ timesteps in
    I_node1 = sim.results.I[:, 1]
    max_idx1 = np.argmax(I_node1)
    assert 20 < max_idx1 < 35, "Node 1 should peak between 20-35 timesteps after the start."

    # Assert: Node 1 infections should decay to zero after peaking and then stay at zero
    zeros_after_peak = np.where(I_node1[max_idx1:] == 0)[0]
    assert zeros_after_peak.size > 0, "Node 1 infections should eventually reach zero after peaking."
    first_zero_idx = max_idx1 + zeros_after_peak[0]
    assert np.all(I_node1[first_zero_idx:] == 0), "Node 1 infections should remain zero after first reaching zero post-peak."

    # Assert: Node 2 infections should peak after Node 1 and then decay to zero
    I_node2 = sim.results.I[:, 2]
    max_idx2 = np.argmax(I_node2)
    assert max_idx2 > max_idx1, "Node 2 should peak after Node 1."
    zeros_after_peak = np.where(I_node2[max_idx2:] == 0)[0]
    assert zeros_after_peak.size > 0, "Node 2 infections should eventually reach zero after peaking."
    first_zero_idx = max_idx2 + zeros_after_peak[0]
    assert np.all(I_node2[first_zero_idx:] == 0), "Node 2 infections should remain zero after first reaching zero post-peak."

    # Assert: Node 3 infections should peak after Node 2 and then decay to zero
    I_node3 = sim.results.I[:, 3]
    max_idx3 = np.argmax(I_node3)
    assert max_idx3 > max_idx2, "Node 3 should peak after Node 2."
    zeros_after_peak = np.where(I_node3[max_idx3:] == 0)[0]
    assert zeros_after_peak.size > 0, "Node 3 infections should eventually reach zero after peaking."
    # TODO - fix this assertion
    # first_zero_idx = max_idx3 + zeros_after_peak[0]
    # assert np.all(I_node3[first_zero_idx:] == 0), "Node 3 infections should remain zero after first reaching zero post-peak."

    return


def test_zero_inflation():
    """
    Test that zero inflation reduces the number of nodes with cases.
    Zero inflation only affects nodes with beta_by_node[i] == 0 (seeded infections).
    """
    n_reps = 20
    duration = 10
    n_nodes = 20

    def setup_zero_inflation_sim(zero_inflation=0.0, seed=None):
        # Set up a scenario where some nodes will have zero local transmission
        # and rely on seeding from neighbors
        init_prev = np.zeros(n_nodes, dtype=np.float32)
        init_prev[0] = 0.3  # Only first node starts with infections

        pars = PropertySet(
            {
                "seed": seed,
                "start_date": lp.date("2020-01-01"),
                "dur": duration,
                "init_pop": np.array([5000] * n_nodes),
                "cbr": np.array([36.5] * n_nodes, dtype=np.float32),
                "r0_scalars": np.ones(n_nodes, dtype=np.float32),
                "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",
                "init_immun": np.array([0.8] * n_nodes, dtype=np.float32),
                "init_prev": init_prev,
                "distances": np.ones((n_nodes, n_nodes)) * 1,  # All nodes 100km apart
                "gravity_k": 5,
                "gravity_a": 5,
                "gravity_b": 5,
                "gravity_c": 1.0,
                "max_migr_frac": 1.0,
                "node_seeding_zero_inflation": zero_inflation,
                "node_seeding_dispersion": 10,  # Lower dispersion for more variability
            }
        )

        sim = lp.SEIR_ABM(pars)
        sim.components = [lp.DiseaseState_ABM, lp.Transmission_ABM, lp.VitalDynamics_ABM]
        return sim

    # Run simulations with different zero inflation values
    seeds = np.arange(1000, 1000 + n_reps)  # Fixed seeds for reproducibility

    nodes_with_cases_no_inflation = []
    nodes_with_cases_inflation = []
    day1_nodes_with_cases_no_inflation = []
    day1_nodes_with_cases_inflation = []

    for seed in seeds:
        # No zero inflation
        sim_no_inflation = setup_zero_inflation_sim(zero_inflation=0.0, seed=seed)
        sim_no_inflation.run()
        total_cases_by_node_no_inflation = sim_no_inflation.results.E[1:].sum(axis=0)  # Sum over time, keep nodes separate
        nodes_with_cases_no_inflation.append(np.sum(total_cases_by_node_no_inflation > 0))  # Number of nodes with cases across all time
        day1_nodes_with_cases_no_inflation.append(np.sum(sim_no_inflation.results.E[1] > 0))  # Number of nodes with cases on day 1

        # 90% zero inflation
        sim_inflation = setup_zero_inflation_sim(zero_inflation=0.9, seed=seed)
        sim_inflation.run()
        total_cases_by_node_inflation = sim_inflation.results.E[1:].sum(axis=0)  # Sum over time, keep nodes separate
        nodes_with_cases_inflation.append(np.sum(total_cases_by_node_inflation > 0))  # Number of nodes with cases across all time
        day1_nodes_with_cases_inflation.append(np.sum(sim_inflation.results.E[1] > 0))  # Number of nodes with cases on day 1

    nodes_with_cases_no_inflation = np.array(nodes_with_cases_no_inflation)
    nodes_with_cases_inflation = np.array(nodes_with_cases_inflation)
    day1_nodes_with_cases_no_inflation = np.array(day1_nodes_with_cases_no_inflation)
    day1_nodes_with_cases_inflation = np.array(day1_nodes_with_cases_inflation)

    # --- Assertions ---
    # Both should have at least node 0 with cases (the initially infected node)
    assert np.all(nodes_with_cases_no_inflation >= 1), "No inflation simulations should always have at least 1 node with cases"
    assert np.all(nodes_with_cases_inflation >= 1), "Inflation simulations should always have at least 1 node with cases"

    # With zero inflation, we should generally have fewer nodes with cases
    mean_nodes_no_inflation_day1 = np.mean(day1_nodes_with_cases_no_inflation)
    mean_nodes_inflation_day1 = np.mean(day1_nodes_with_cases_inflation)
    assert mean_nodes_inflation_day1 < mean_nodes_no_inflation_day1, (
        f"Zero inflation should reduce nodes with cases. No inflation: {mean_nodes_no_inflation_day1:.2f}, With inflation: {mean_nodes_inflation_day1:.2f}"
    )
    mean_nodes_no_inflation = np.mean(nodes_with_cases_no_inflation)
    mean_nodes_inflation = np.mean(nodes_with_cases_inflation)
    assert mean_nodes_inflation < mean_nodes_no_inflation, (
        f"Zero inflation should reduce nodes with cases. No inflation: {mean_nodes_no_inflation:.2f}, With inflation: {mean_nodes_inflation:.2f}"
    )

    # Test extreme case: 99% zero inflation should dramatically reduce spread
    nodes_with_cases_extreme_inflation = []
    day1_nodes_with_cases_extreme_inflation = []
    for seed in seeds[:5]:  # Fewer reps for extreme case
        sim_extreme_inflation = setup_zero_inflation_sim(zero_inflation=0.99, seed=seed)
        sim_extreme_inflation.run()
        total_cases_by_node_extreme = sim_extreme_inflation.results.E[1:].sum(axis=0)
        nodes_with_cases_extreme_inflation.append(np.sum(total_cases_by_node_extreme > 0))
        day1_nodes_with_cases_extreme_inflation.append(np.sum(sim_extreme_inflation.results.E[1] > 0))

    mean_nodes_extreme_inflation = np.mean(nodes_with_cases_extreme_inflation)
    mean_nodes_extreme_inflation_day1 = np.mean(day1_nodes_with_cases_extreme_inflation)
    assert mean_nodes_extreme_inflation_day1 < mean_nodes_inflation_day1, (
        f"Extreme zero inflation should reduce spread even more. No inflation: {mean_nodes_inflation_day1:.2f}, Extreme: {mean_nodes_extreme_inflation_day1:.2f}"
    )
    assert mean_nodes_extreme_inflation < mean_nodes_inflation, (
        f"Extreme zero inflation should reduce spread even more. No inflation: {mean_nodes_inflation:.2f}, Extreme: {mean_nodes_extreme_inflation:.2f}"
    )


@patch("laser_polio.root", Path("tests/"))
def test_r0_sans_heterogeneity():
    """
    Test that r0 generates the expected number of infections withOUT heterogeneity.
    E.g., R0 = 14 should generate ~14 infections.

    Assumes:
    - No immunity
    - No heterogeneity
    - R0 spatial scalars are all 1.0
    - No seasonality
    - No births or deaths
    - No routine immunization
    - No SIAs
    """

    # Key assumptions
    init_immun_scalar = 0.0
    individual_heterogeneity = False
    r0_scalar_wt_slope = 0.0  # ensures that r0_scalars = 1.0
    r0_scalar_wt_intercept = 0.5  # ensures that r0_scalars = 1.0
    seasonal_amplitude = 0.0  # no seasonality
    cbr = np.array([0])  # no births or deaths
    vx_prob_ri = None  # no routine immunization
    vx_prob_sia = None  # no SIA
    ipv_vx = False
    n_days = 30
    dur_inf = lp.constant(value=25)  # Single infection will expire before end of sim
    dur_exp = lp.constant(value=60)  # Long exposures ensure that exposed individuals will be in the E state at the end of the simulation
    n_reps = 10

    # Setting pars
    r0 = 14
    regions = ["ZAMFARA"]
    start_year = 2018
    pop_scale = 1 / 1
    init_region = "ANKA"
    init_prev = 1
    migration_method = "radiation"
    radiation_k_log10 = -0.3
    max_migr_frac = 0.1
    verbose = 0
    save_plots = False
    save_data = False
    plot_pars = False
    results_path = "results/test_r0_sans_heterogeneity"
    init_pop_file = None
    use_pim_scalars = True

    Es = []
    Is = []
    for _rep in range(n_reps):
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
            r0=r0,
            migration_method=migration_method,
            radiation_k_log10=radiation_k_log10,
            max_migr_frac=max_migr_frac,
            init_pop_file=init_pop_file,
            use_pim_scalars=use_pim_scalars,
            individual_heterogeneity=individual_heterogeneity,
            init_immun_scalar=init_immun_scalar,
            r0_scalar_wt_slope=r0_scalar_wt_slope,
            r0_scalar_wt_intercept=r0_scalar_wt_intercept,
            seasonal_amplitude=seasonal_amplitude,
            cbr=cbr,
            vx_prob_ri=vx_prob_ri,
            vx_prob_sia=vx_prob_sia,
            ipv_vx=ipv_vx,
            dur_exp=dur_exp,
            dur_inf=dur_inf,
        )

        E = np.sum(sim.results.E_by_strain[:, :, 0], axis=1)  # Filter to VDPV2 strain & sum over nodes
        Es.append(E)
        I = np.sum(sim.results.I_by_strain[:, :, 0], axis=1)  # Filter to VDPV2 strain & sum over nodes
        Is.append(I)

    I_init = np.array([x[0] for x in Is])
    I_final = np.array([x[-1] for x in Is])
    E_final = np.array([x[-1] for x in Es])
    assert np.all(I_init == 1), f"There should be one infection at the start of the simulation, but got {I_init}."
    assert np.all(I_final == 0), f"There should be no infections at the end of the simulation, but got {I_final}."
    assert np.isclose(np.mean(E_final), 14, atol=7), (
        f"There should be approximately 14 exposures at the end of the simulation, but got {np.mean(E_final)}."
    )


def test_r0_with_heterogeneity():
    """
    Test that r0 generates the expected number of infections WITH heterogeneity.
    E.g., R0 = 14 should generate ~14 infections.

    Assumes:
    - No immunity
    - WITH heterogeneity
    - R0 spatial scalars are all 1.0
    - No seasonality
    - No births or deaths
    - No routine immunization
    - No SIAs
    """

    # Key assumptions
    init_immun_scalar = 0.0
    individual_heterogeneity = True
    r0_scalar_wt_slope = 0.0  # ensures that r0_scalars = 1.0
    r0_scalar_wt_intercept = 0.5  # ensures that r0_scalars = 1.0
    seasonal_amplitude = 0.0  # no seasonality
    cbr = np.array([0])  # no births or deaths
    vx_prob_ri = None  # no routine immunization
    vx_prob_sia = None  # no SIA
    ipv_vx = False
    n_days = 30
    dur_inf = lp.constant(value=25)  # Single infection will expire before end of sim
    dur_exp = lp.constant(value=60)  # Long exposures ensure that exposed individuals will be in the E state at the end of the simulation
    n_reps = 10

    # Setting pars
    r0 = 14
    regions = ["ZAMFARA"]
    start_year = 2018
    pop_scale = 1 / 1
    init_region = "ANKA"
    init_prev = 1
    migration_method = "radiation"
    radiation_k_log10 = -0.3
    max_migr_frac = 0.1
    verbose = 0
    save_plots = False
    save_data = False
    plot_pars = False
    results_path = "results/test_r0_sans_heterogeneity"
    init_pop_file = None
    use_pim_scalars = True

    Es = []
    Is = []
    risks = []
    infectivities = []
    for _rep in range(n_reps):
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
            r0=r0,
            migration_method=migration_method,
            radiation_k_log10=radiation_k_log10,
            max_migr_frac=max_migr_frac,
            init_pop_file=init_pop_file,
            use_pim_scalars=use_pim_scalars,
            individual_heterogeneity=individual_heterogeneity,
            init_immun_scalar=init_immun_scalar,
            r0_scalar_wt_slope=r0_scalar_wt_slope,
            r0_scalar_wt_intercept=r0_scalar_wt_intercept,
            seasonal_amplitude=seasonal_amplitude,
            cbr=cbr,
            vx_prob_ri=vx_prob_ri,
            vx_prob_sia=vx_prob_sia,
            ipv_vx=ipv_vx,
            dur_exp=dur_exp,
            dur_inf=dur_inf,
        )

        E = np.sum(sim.results.E_by_strain[:, :, 0], axis=1)  # Filter to VDPV2 strain & sum over nodes
        Es.append(E)
        I = np.sum(sim.results.I_by_strain[:, :, 0], axis=1)  # Filter to VDPV2 strain & sum over nodes
        Is.append(I)

        # Check individual heterogeneity
        risk = sim.people.acq_risk_multiplier
        risks.append(risk)
        infectivity = sim.people.daily_infectivity
        infectivities.append(infectivity)

    I_init = np.array([x[0] for x in Is])
    I_final = np.array([x[-1] for x in Is])
    E_final = np.array([x[-1] for x in Es])
    assert np.all(I_init == 1), f"There should be one infection at the start of the simulation, but got {I_init}."
    assert np.all(I_final == 0), f"There should be no infections at the end of the simulation, but got {I_final}."
    assert np.isclose(np.mean(E_final), 14, atol=10), (
        f"There should be approximately 14 exposures at the end of the simulation, but got {np.mean(E_final)}."
    )

    # Pool all risk values across reps for more statistical power
    pooled_risks = np.concatenate(risks)
    pooled_infectivities = np.concatenate(infectivities)

    # Test 1: Mean risk should be approximately 1.0 (lognormal with mean=1.0 by design)
    assert np.isclose(np.mean(pooled_risks), 1.0, rtol=0.1), f"Risk multiplier mean should be ~1.0, but got {np.mean(pooled_risks):.3f}"

    # Test 2: Right skewness - mean > median for right-skewed distributions
    risk_mean = np.mean(pooled_risks)
    risk_median = np.median(pooled_risks)
    assert risk_mean > risk_median, (
        f"Risk values should be right-skewed (mean > median), but got mean={risk_mean:.3f}, median={risk_median:.3f}"
    )

    # Test 3: Kolmogorov-Smirnov test for lognormality
    # Fit lognormal to the data
    log_risks = np.log(pooled_risks[pooled_risks > 0])  # Remove any zeros
    log_mean = np.mean(log_risks)
    log_std = np.std(log_risks)
    # Test if log-transformed data is approximately normal
    ks_stat, p_value = stats.kstest(log_risks, lambda x: stats.norm.cdf(x, loc=log_mean, scale=log_std))
    assert p_value > 0.01, (  # Don't reject normality of log(risk) at 1% level
        f"Log-transformed risks should be approximately normal (lognormal test), but KS test p-value={p_value:.4f}"
    )

    # Test 4: Variance should be approximately equal to risk_mult_var parameter (4.0)
    assert np.isclose(np.var(pooled_risks), 4.0, rtol=0.3), f"Risk variance should be ~4.0, but got {np.var(pooled_risks):.3f}"

    # Test 5: Correlation between risk and infectivity should be approximately 0.8
    pooled_corr = spearmanr(pooled_risks, pooled_infectivities).correlation
    assert np.isclose(pooled_corr, 0.8, atol=0.1), f"Risk-infectivity correlation should be ~0.8, but got {pooled_corr:.3f}"

    # Test 6: Mean infectivity should be approximately 14/24
    assert np.isclose(np.mean(pooled_infectivities), 14 / 24, atol=0.1), (
        f"Infectivity mean should be ~14/24, but got {np.mean(pooled_infectivities):.3f}"
    )

    # Test 7: Infectivity should be right-skewed
    infectivity_mean = np.mean(pooled_infectivities)
    infectivity_median = np.median(pooled_infectivities)
    assert infectivity_mean > infectivity_median, (
        f"Infectivity values should be right-skewed (mean > median), but got mean={infectivity_mean:.3f}, median={infectivity_median:.3f}"
    )


if __name__ == "__main__":
    # test_trans_default()
    # test_zero_trans()
    # test_double_trans()
    # test_linear_transmission()
    # test_zero_inflation()
    test_r0_sans_heterogeneity()
    test_r0_with_heterogeneity()
    print("All transmission tests passed!")
