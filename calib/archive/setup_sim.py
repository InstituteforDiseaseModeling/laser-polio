import json
from pathlib import Path

import numpy as np
import pandas as pd
import sciris as sc
from laser_core.propertyset import PropertySet

import laser_polio as lp

"""
This script contains a demo simulation of polio transmission in Nigeria.

The model uses the same data and setup as the EMOD model, except in the following instances:
- The model assumes everyone >15y is immune
- I'm using a sinusoidal seasonality function rather than a step function
- The nodes are not divided below the adm2 level (with no plans to do so)
- There is no scaling of transmission between N & S Nigeria (other than underweight fraction)
- We do not update the cbr, ri, sia, or underwt data over time
- Vaccines are not allowed to transmit
"""


def setup_sim(config=None, **kwargs):
    """Set up simulation from config file (YAML) or kwargs."""

    # Fallback for direct function calls
    config = config or {}

    # Unpack either from config dict or fallback to keyword args
    regions = config.get("regions", kwargs.get("regions", ["ZAMFARA"]))
    start_year = config.get("start_year", kwargs.get("start_year", 2019))
    n_days = config.get("n_days", kwargs.get("n_days", 365))
    pop_scale = config.get("pop_scale", kwargs.get("pop_scale", 1 / 10))
    init_region = config.get("init_region", kwargs.get("init_region", "ANKA"))
    init_prev = config.get("init_prev", kwargs.get("init_prev", 0.001))
    results_path = config.get("results_path", kwargs.get("results_path", "results/demo"))
    save_plots = config.get("save_plots", kwargs.get("save_plots", False))
    save_data = config.get("save_data", kwargs.get("save_data", False))

    # Find the dot_names matching the specified string(s)
    dot_names = lp.find_matching_dot_names(regions, lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")

    # Load the shape names and centroids (sans geometry)
    centroids = pd.read_csv(lp.root / "data/shp_names_africa_adm2.csv")
    centroids = centroids.set_index("dot_name").loc[dot_names]

    # Initial immunity
    init_immun = pd.read_hdf(lp.root / "data/init_immunity_0.5coverage_january.h5", key="immunity")
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
    dist_matrix = lp.get_distance_matrix(lp.root / "data/distance_matrix_africa_adm2.h5", dot_names)  # Load distances matrix (km)

    # SIA schedule
    start_date = lp.date(f"{start_year}-01-01")
    historic_sia_schedule = pd.read_csv(lp.root / "data/sia_historic_schedule.csv")
    future_sia_schedule = pd.read_csv(lp.root / "data/sia_scenario_1.csv")
    sia_schedule_raw = pd.concat([historic_sia_schedule, future_sia_schedule], ignore_index=True)  # combine the two schedules
    sia_schedule = lp.process_sia_schedule_polio(sia_schedule_raw, dot_names, start_date)  # Load sia schedule

    ### Load the demographic, coverage, and risk data
    # Age pyramid
    age = pd.read_csv(lp.root / "data/age_africa.csv")
    age = age[(age["adm0_name"] == "NIGERIA") & (age["Year"] == start_year)]
    # Compiled data
    df_comp = pd.read_csv(lp.root / "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
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

    # Assert that all data arrays have the same length
    assert (
        len(dot_names)
        == len(dist_matrix)
        == len(init_immun)
        == len(centroids)
        == len(init_prevs)
        == len(pop)
        == len(cbr)
        == len(ri)
        == len(sia_prob)
        == len(r0_scalars)
    )

    # Set parameters
    pars = PropertySet(
        {
            # Time
            "start_date": start_date,  # Start date of the simulation
            "dur": n_days,  # Number of timesteps
            # Population
            "n_ppl": pop,  # np.array([30000, 10000, 15000, 20000, 25000]),
            "age_pyramid_path": lp.root / "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "cbr": cbr,  # Crude birth rate per 1000 per year
            # Disease
            "init_immun": init_immun,  # Initial immunity per node
            "init_prev": init_prevs,  # Initial prevalence per node (1% infected)
            "r0": 14,  # Basic reproduction number
            "risk_mult_var": 4.0,  # Lognormal variance for the individual-level risk multiplier (risk of acquisition multiplier; mean = 1.0)
            "corr_risk_inf": 0.8,  # Correlation between individual risk multiplier and individual infectivity (daily infectivity, mean = 14/24)
            "r0_scalars": r0_scalars,  # Spatial transmission scalar (multiplied by global rate)
            "seasonal_amplitude": 0.125,  # Seasonal variation in transmission
            "seasonal_peak_doy": 180,  # Phase of seasonal variation
            "p_paralysis": 1 / 2000,  # Probability of paralysis
            "dur_exp": lp.normal(mean=3, std=1),  # Duration of the exposed state
            "dur_inf": lp.gamma(shape=4.51, scale=5.32),  # Duration of the infectious state
            # Migration
            "distances": dist_matrix,  # Distance in km between nodes
            "gravity_k": 0.5,  # Gravity scaling constant
            "gravity_a": 1,  # Origin population exponent
            "gravity_b": 1,  # Destination population exponent
            "gravity_c": 2.0,  # Distance exponent
            "max_migr_frac": 0.01,  # Fraction of population that migrates
            "centroids": centroids,  # Centroids of the nodes
            # Interventions
            "vx_prob_ri": ri,  # Probability of routine vaccination
            "sia_schedule": sia_schedule,  # Schedule of SIAs
            "vx_prob_sia": sia_prob,  # Effectiveness of SIAs
        }
    )

    with Path("params.json").open("r") as par:
        params = json.load(par)
    pars += params

    # Initialize the sim
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.Transmission_ABM, lp.RI_ABM, lp.SIA_ABM]

    # Run the simulation
    sim.run()

    # Plot results
    if save_plots:
        sim.plot(save=True, results_path=results_path)

    if save_data:
        Path(results_path).mkdir(parents=True, exist_ok=True)
        lp.save_results_to_csv(sim.results, filename=results_path + "/simulation_results.csv")

    sc.printcyan("Done.")


if __name__ == "__main__":
    # import argparse
    # import yaml

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-config", type=str, required=True, help="Path to base model config YAML")

    # Set up the simulation parameters and data
    setup_sim()
