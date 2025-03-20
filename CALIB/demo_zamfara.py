import csv
import json
import pickle
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
- The total population counts are being estimated by scaling up u5 population counts based on their proportion of the population
- I'm using a sinusoidal seasonality function rather than a step function
- The nodes are not divided below the adm2 level (with no plans to do so)
- There is no scaling of transmission between N & S Nigeria (other than underweight fraction)
- We do not update the cbr, ri, sia, or underwt data over time
"""

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2019
n_days = 365
pop_scale = 1 / 10
init_region = "ANKA"
init_prev = 0.001
r0 = 200
results_path = "results/demo_zamfara"

######### END OF USER PARS ########
###################################


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


class ConfigurablePropertySet(PropertySet):
    def __init__(self, default_params: dict):
        """
        Initialize a ConfigurablePropertySet with default parameters.

        :param default_params: The dictionary containing the default parameters.
        """
        super().__init__(default_params)  # Pass default params to base class

    def override(self, override_params: dict):
        """
        Override default parameters with values from override_params.

        :param override_params: The dictionary containing override parameters (loaded from disk).
        :raises ValueError: If override_params contains unknown keys.
        """
        # Fix: Use to_dict() to retrieve keys from the PropertySet
        unexpected_keys = set(override_params.keys()) - set(self.to_dict().keys())
        if unexpected_keys:
            raise ValueError(f"Unexpected parameters in override set: {unexpected_keys}")

        # Apply overrides to the PropertySet
        for key, value in override_params.items():
            self[key] = value  # Assuming PropertySet supports item assignment

    def save(self, filename: str = "params.pkl"):
        """
        Save the current property set to a pickle file.

        :param filename: Path to the output pickle file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.to_dict(), f)

        print(f"Configuration saved to {filename}")


class ConfigurablePropertySetPrev(PropertySet):
    def __init__(self, default_params: dict):
        """
        Initialize a ConfigurablePropertySet with default parameters.

        :param default_params: The dictionary containing the default parameters.
        """
        super().__init__(default_params)  # Pass default params to base class

    def override(self, override_params: dict):
        """
        Override default parameters with values from override_params.

        :param override_params: The dictionary containing override parameters (loaded from disk).
        :raises ValueError: If override_params contains unknown keys.
        """
        unexpected_keys = set(override_params.keys()) - set(self.keys())
        if unexpected_keys:
            raise ValueError(f"Unexpected parameters in override set: {unexpected_keys}")

        # Apply overrides to the PropertySet
        for key, value in override_params.items():
            self[key] = value  # Assuming PropertySet supports item assignment

    def save(self, filename: str = "params.pkl"):
        """
        Save the current property set to a JSON file.

        :param filename: Path to the output JSON file.
        """
        # with open(filename, "w") as f:
        #    json.dump(self.to_dict(), f, indent=4)

        with open("params_final.pkl", "wb") as f:
            pickle.dump(self.to_dict(), f)

        print(f"Configuration saved to {filename}")


# Set parameters
pars = ConfigurablePropertySet(
    {
        # Time
        "start_date": start_date,  # Start date of the simulation
        "dur": n_days,  # Number of timesteps
        # Population
        "n_ppl": pop,  # np.array([30000, 10000, 15000, 20000, 25000]),
        "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
        "cbr": cbr,  # np.array([37, 41, 30, 25, 33]),  # Crude birth rate per 1000 per year
        # Disease
        "init_immun": init_immun,  # Initial immunity per node
        "init_prev": init_prevs,  # Initial prevalence per node (1% infected)
        "r0": r0,  # Basic reproduction number
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
        "sia_eff": sia,  # Effectiveness of SIAs
        "life_expectancies": np.ones(len(dot_names)) * 65,  # placeholder, should probably derive from age pyramid
    }
)

with Path("params.json").open("r") as par:
    params = json.load(par)

pars.override(params)

# Initialize the sim
sim = lp.SEIR_ABM(pars)
sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.Transmission_ABM, lp.RI_ABM, lp.SIA_ABM]

# Run the simulation
sim.run()


def save_results_to_csv(results, filename="simulation_results.csv"):
    """
    Save simulation results (S, E, I, R) to a CSV file with columns: Time, Node, S, E, I, R.

    :param results: The results object containing numpy arrays for S, E, I, and R.
    :param filename: The name of the CSV file to save.
    """
    timesteps, nodes = results.S.shape  # Get the number of timesteps and nodes

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["Time", "Node", "S", "E", "I", "R"])

        # Write data
        for t in range(timesteps):
            for n in range(nodes):
                writer.writerow([t, n, results.S[t, n], results.E[t, n], results.I[t, n], results.R[t, n]])

    print(f"Results saved to {filename}")


# Example usage
save_results_to_csv(sim.results)

# # Plot results
# sim.plot(save=True, results_path=results_path)

sc.printcyan("Done.")
