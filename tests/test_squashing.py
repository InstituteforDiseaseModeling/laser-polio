import numpy as np
import pandas as pd
from laser_core.propertyset import PropertySet

import laser_polio as lp

"""
This test is based on examples/demo_nigeria.py
"""

regions = ["ZAMFARA"]
start_year = 2019
n_days = 5
pop_scale = 1 / 100
init_region = "ANKA"
init_prev = 0.01

# Find the dot_names matching the specified string(s)
dot_names = lp.find_matching_dot_names(regions, "data/compiled_cbr_pop_ri_sia_underwt_africa.csv")

# Load the shape names and centroids (sans geometry)
centroids = pd.read_csv("data/shp_names_africa_adm2.csv")
centroids = centroids.set_index("dot_name").loc[dot_names]

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
# TODO make sure this is the same order as the dot_names
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
age = age[(age["ADM0_NAME"] == "NIGERIA") & (age["Year"] == start_year)]
prop_u5 = age.loc[age["age_group"] == "0-4", "population"].values[0] / age["population"].sum()
# Compiled data
df_comp = pd.read_csv("data/compiled_cbr_pop_ri_sia_underwt_africa.csv")
df_comp = df_comp[df_comp["year"] == start_year]
# Population data
pop_u5 = df_comp.set_index("dot_name").loc[dot_names, "pop_u5"].values  # Extract the pop data in the same order as the dot_names
pop = pop_u5 / prop_u5  # Estimate the total population size since the data is only for under 5s
pop = pop * pop_scale  # Scale population
cbr = df_comp.set_index("dot_name").loc[dot_names, "cbr"].values  # CBR data
ri = df_comp.set_index("dot_name").loc[dot_names, "ri_eff"].values  # RI data
sia = df_comp.set_index("dot_name").loc[dot_names, "sia_prob"].values  # SIA data
beta_spatial = df_comp.set_index("dot_name").loc[dot_names, "underwt_prop"].values  # Underweight data

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
    == len(sia)
    == len(beta_spatial)
)


def setup_sim():
    """Initialize a test simulation with DiseaseState_ABM component."""
    pars = PropertySet(
        {
            # Time
            "start_date": start_date,  # Start date of the simulation
            "dur": 30,  # Number of timesteps
            # Population
            "n_ppl": pop,  # np.array([30000, 10000, 15000, 20000, 25000]),
            "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
            "cbr": cbr,  # np.array([37, 41, 30, 25, 33]),  # Crude birth rate per 1000 per year
            # Disease
            "init_immun": init_immun,  # Initial immunity per node
            "init_prev": init_prevs,  # Initial prevalence per node (1% infected)
            "r0": 14,  # Basic reproduction number
            "risk_mult_var": 4.0,  # Lognormal variance for the individual-level risk multiplier (risk of acquisition multiplier; mean = 1.0)
            "corr_risk_inf": 0.8,  # Correlation between individual risk multiplier and individual infectivity (daily infectivity, mean = 14/24)
            "beta_spatial": beta_spatial,  # Spatial transmission scalar (multiplied by global rate)
            "seasonal_factor": 0.125,  # Seasonal variation in transmission
            "seasonal_phase": 180,  # Phase of seasonal variation
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
            "vx_prob_sia": sia,  # Effectiveness of SIAs
            "life_expectancies": np.ones(len(dot_names)) * 65,  # placeholder, should probably derive from age pyramid
        }
    )
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.Transmission_ABM]
    return sim


# Test Initialization
def test_squash():
    """Ensure disease state properties are correctly initialized."""
    sim = setup_sim()

    age = sim.people.date_of_birth[: sim.people.count] * -1
    assert np.all(age / 365 <= 15), "No >15yo should be present in the sim since we're assuming they're Rs."

    disease_state = sim.people.disease_state[: sim.people.count]
    assert np.all(disease_state < 3), "No one should be in the recovered state since they should be squashed out."

    exp_pop = np.sum(pop)
    obs_pop = sim.people.count + sim.results.R[0].sum()
    assert np.isclose(exp_pop, obs_pop, atol=100), f"Expected population size {exp_pop}, but got {obs_pop}."

    # Check the number of recovered in results. I did a really rough calc for <15yo immunity since it's broken up by age bins
    exp_o15 = exp_pop * 0.43
    exclude_cols = ["guid", "period", "serotype", "dpt1", "dpt3"]
    mean_immun = np.mean(init_immun.drop(columns=exclude_cols))
    exp_u15 = exp_pop - exp_o15
    exp_u15_r = exp_u15 * mean_immun
    exp_r = exp_o15 + exp_u15_r
    obs_r = sim.results.R[0].sum()
    assert np.isclose(exp_r, obs_r, atol=4000), f"Expected results.R size {exp_r}, but got {obs_r}."

    init_R = np.sum(sim.results.R[-1])

    # Run the simulation
    sim.run()

    # Ensure that the number of R increased due to transmission
    end_R = np.sum(sim.results.R[-1])
    assert end_R > init_R, "The number of recovered individuals should have increased after running the simulation."


if __name__ == "__main__":
    test_squash()
    print("All squash tests passed!")
