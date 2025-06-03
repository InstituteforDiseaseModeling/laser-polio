import pickle

import matplotlib.pyplot as plt
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
- Vaccines are not allowed to transmit
"""

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_year = 2019
n_days = 365
pop_scale = 1 / 1
init_region = "ANKA"
init_prev = 0.001
r0 = 200
results_path = "results/demo_zamfara"

######### END OF USER PARS ########
###################################


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


# Assert that all data arrays have the same length
assert (
    len(dot_names)
    == len(dist_matrix)
    == len(init_immun)
    == len(node_lookup)
    == len(init_prevs)
    == len(pop)
    == len(cbr)
    == len(ri)
    == len(sia_prob)
    == len(r0_scalars)
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
        "node_lookup": node_lookup,  # Node info (node_id are keys, dict contains dot_name, lat, lon)
        # Interventions
        "vx_prob_ri": ri,  # Probability of routine vaccination
        "sia_schedule": sia_schedule,  # Schedule of SIAs
        "vx_prob_sia": sia_prob,  # Effectiveness of SIAs
    }
)

sc.printcyan("Done.")


# Define the range of r0 values to sweep
r0_values = np.linspace(0, 200, 5)  # You can adjust how many points to simulate
total_infected_by_r0 = []

for r0 in r0_values:
    print(f"Running sim for R0 = {r0:.2f}")

    # Set parameters for this r0
    pars["r0"] = r0
    sim = lp.SEIR_ABM(pars)
    sim.components = [lp.VitalDynamics_ABM, lp.DiseaseState_ABM, lp.Transmission_ABM, lp.RI_ABM, lp.SIA_ABM]
    sim.run()

    # Calculate total infected across all time and nodes
    total_infected = sim.results.I.sum()
    total_infected_by_r0.append(total_infected)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(r0_values, total_infected_by_r0, marker="o")
plt.xlabel("R₀")
plt.ylabel("Total Infected (I)")
plt.title("Total Infected vs R₀")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/total_infected_vs_r0.png")
plt.show()

sc.printcyan("Done.")
