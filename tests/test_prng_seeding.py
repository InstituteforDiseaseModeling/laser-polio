import unittest

import numpy as np
import pandas as pd
from laser_core.propertyset import PropertySet

import laser_polio as lp


class TestPRNGSeeding(unittest.TestCase):
    def test_same_seed(self):
        sim1 = setup_sim(seed=42)
        sim1.run()

        # Setup the second simulation with the same seed _after_ sim1.run()
        # otherwise sim1.run() will invalidate setting the global seed.
        sim2 = setup_sim(seed=42)
        sim2.run()

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

        return

    def test_different_seeds(self):
        sim1 = setup_sim(seed=42)
        sim1.run()

        # Setup the second simulation with a different seed _after_ sim1.run()
        # otherwise sim1.run() will invalidate setting the global seed.
        sim2 = setup_sim(seed=13)
        sim2.run()

        # assert False, "The two simulations should be identical, but they are not."
        same = (
            np.all(sim1.results.births == sim2.results.births)
            and np.all(sim1.results.deaths == sim2.results.deaths)
            and np.all(sim1.results.paralyzed == sim2.results.paralyzed)
            and np.all(sim1.results.ri_vaccinated == sim2.results.ri_vaccinated)
            and np.all(sim1.results.sia_protected == sim2.results.sia_protected)
            and np.all(sim1.results.sia_vaccinated == sim2.results.sia_vaccinated)
        )
        assert not same, "The two simulations should be different, but they are not."

        return


def setup_sim(seed):
    regions = ["ZAMFARA"]
    start_year = 2019
    n_days = 365
    pop_scale = 1 / 100
    init_region = "ANKA"
    init_prev = 0.02  # up from 0.001
    r0 = 14

    # Find the dot_names matching the specified string(s)
    dot_names = lp.find_matching_dot_names(regions, "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")

    # Load the node_lookup dictionary with node_id, dot_names, centroids
    node_lookup = lp.get_node_lookup("data/node_lookup.json", dot_names)

    # Initial immunity
    init_immun = pd.read_hdf("data/init_immunity_0.5coverage_january.h5", key="immunity")
    init_immun = init_immun.set_index("dot_name").loc[dot_names]
    init_immun = init_immun[init_immun["period"] == start_year]

    # Initial prevalence
    init_prevs = np.zeros(len(dot_names))
    prev_indices = [i for i, dot_name in enumerate(dot_names) if init_region in dot_name]
    print(f"Infections will be seeded in {len(prev_indices)} nodes containing the string {init_region} at {init_prev} prevalence.")
    # Throw an error if the region is not found
    if len(prev_indices) == 0:
        raise ValueError(f"No nodes found containing the string {init_region}. Cannot seed infections.")
    init_prevs[prev_indices] = init_prev

    # Distance matrix
    dist_matrix = lp.get_distance_matrix("data/distance_matrix_africa_adm2.h5", dot_names)  # Load distances matrix (km)

    # SIA schedule
    start_date = lp.date(f"{start_year}-01-01")
    historic_sia_schedule = pd.read_csv("data/sia_historic_schedule.csv")
    future_sia_schedule = pd.read_csv("data/sia_scenario_1.csv")
    sia_schedule_raw = pd.concat([historic_sia_schedule, future_sia_schedule], ignore_index=True)  # combine the two schedules
    sia_schedule = lp.process_sia_schedule_polio(sia_schedule_raw, dot_names, start_date)  # Load sia schedule

    ### Load the demographic, coverage, and risk data
    # Age pyramid
    age = pd.read_csv("data/age_africa.csv")
    age = age[(age["adm0_name"] == "NIGERIA") & (age["Year"] == start_year)]
    # Compiled data
    df_comp = pd.read_csv("data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
    df_comp = df_comp[df_comp["year"] == start_year]
    # Population data
    pop = df_comp.set_index("dot_name").loc[dot_names, "pop_total"].values  # total population (all ages)
    pop = pop * pop_scale  # Scale population
    cbr = df_comp.set_index("dot_name").loc[dot_names, "cbr"].values  # CBR data
    ri = df_comp.set_index("dot_name").loc[dot_names, "ri_eff"].values  # RI data
    sia_re = df_comp.set_index("dot_name").loc[dot_names, "sia_random_effect"].values  # SIA data
    sia_prob = lp.calc_sia_prob_from_rand_eff(sia_re, center=0.7, scale=2.4)  # Secret sauce numbers from Hil
    reff_re = df_comp.set_index("dot_name").loc[dot_names, "reff_random_effect"].values  # random effects from regression model
    r0_scalars = lp.calc_r0_scalars_from_rand_eff(reff_re)  # Center and scale the random effects

    pars = PropertySet(
        {
            # PRNG seed
            "seed": seed,  # Random number generator seed
            # Time
            "start_date": start_date,  # Start date of the simulation
            "dur": n_days,  # Number of timesteps
            # Population
            "n_ppl": pop,  # np.array([30000, 10000, 15000, 20000, 25000]),
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "cbr": cbr,  # Crude birth rate per 1000 per year
            # Disease
            "init_immun": init_immun,  # Initial immunity per node
            "init_prev": init_prevs,  # Initial prevalence per node (1% infected)
            "r0": r0,  # Basic reproduction number
            "risk_mult_var": 4.0,  # Lognormal variance for the individual-level risk multiplier (risk of acquisition multiplier; mean = 1.0)
            "corr_risk_inf": 0.8,  # Correlation between individual risk multiplier and individual infectivity (daily infectivity, mean = 14/24)
            "r0_scalars": r0_scalars,  # Spatial transmission scalar (multiplied by global rate)
            "seasonal_factor": 0.125,  # Seasonal variation in transmission
            "seasonal_phase": 180,  # Phase of seasonal variation
            "p_paralysis": 1 / 50,  # Probability of paralysis (up from 1 / 2000)
            "dur_exp": lp.normal(mean=3, std=1),  # Duration of the exposed state
            "dur_inf": lp.gamma(shape=4.51, scale=5.32),  # Duration of the infectious state
            # Migration
            "distances": dist_matrix,  # Distance in km between nodes
            "gravity_k": 0.5,  # Gravity scaling constant
            "gravity_a": 1,  # Origin population exponent
            "gravity_b": 1,  # Destination population exponent
            "gravity_c": 2.0,  # Distance exponent
            "max_migr_frac": 0.05,  # Fraction of population that migrates (up from 0.01)
            "node_lookup": node_lookup,  # Node info (node_id are keys, dict contains dot_name, lat, lon)
            # Interventions
            "vx_prob_ri": ri,  # Probability of routine vaccination
            "sia_schedule": sia_schedule,  # Schedule of SIAs
            "vx_prob_sia": sia_prob,  # Effectiveness of SIAs
        }
    )

    # Initialize the sim
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.Transmission_ABM, lp.RI_ABM, lp.SIA_ABM]

    return sim


if __name__ == "__main__":
    unittest.main()
