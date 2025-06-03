import numpy as np
import pandas as pd
import sc
from laser_core.propertyset import PropertySet

import laser_polio as lp
from laser_polio.utils import get_tot_pop_and_cbr
from laser_polio.utils import process_sia_schedule

# Set the start date
start_date = sc.date("2025-01-01")

# Load population & CBR data from the UN WPP dataset
isos = ["BEN", "BFA", "CIV", "GHA", "TGO"]  # ["Benin", "Burkina Faso", "Cote dIvoire", "Ghana",  "Togo"]
pop, cbr = get_tot_pop_and_cbr("data/unwpp.csv", isos=isos, year=2023)
pop = pop / 1000  # Scale down population for speed

# Load distances between countries
dist_df = pd.read_csv("data/distance_matrix_country_iso.csv", index_col=0)
dist_df_filtered = dist_df.loc[isos, isos]
dist_matrix = dist_df_filtered.values

# Load sia schedule
sia_schedule = process_sia_schedule("data/sia_schedule.csv", start_date)


# Set parameters
pars = PropertySet(
    {
        # Time
        "start_date": start_date,  # Start date of the simulation
        "dur": 180,  # Number of timesteps
        # Population
        "n_ppl": pop,  # np.array([30000, 10000, 15000, 20000, 25000]),
        "distances": dist_matrix,  # Distance in km
        # Disease
        "init_prev": np.array([0, 0.01, 0, 0, 0]),  # Initial prevalence per node (1% infected)
        "beta_global": 0.3,  # Global infection rate
        "r0_scalars": np.array([0.8, 2.0, 0.9, 1.5, 0.5]),  # Spatial transmission scalar (multiplied by global rate)
        "seasonal_amplitude": 0.125,  # Seasonal variation in transmission
        "seasonal_peak_doy": 180,  # Phase of seasonal variation
        "p_paralysis": 1 / 20,  # Probability of paralysis
        "dur_exp": 4,  # Duration of the exposed state
        "dur_inf": 24,  # Duration of the infectious state
        # Migration
        "gravity_k": 1,  # Gravity scaling constant
        "gravity_a": 1,  # Origin population exponent
        "gravity_b": 1,  # Destination population exponent
        "gravity_c": 2.0,  # Distance exponent
        "migration_frac": 0.01,  # Fraction of population that migrates
        # Demographics & vital dynamics
        "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
        "cbr": cbr,  # np.array([37, 41, 30, 25, 33]),  # Crude birth rate per 1000 per year
        # Interventions
        "vx_prob_ri": np.array([0.1, 0.5, 0.01, 0, 0.2]),  # Probability of routine vaccination
        "sia_schedule": sia_schedule,  # Schedule of SIAs
    }
)

# Initialize the sim
sim = lp.SEIR_ABM(pars)
sim.add_component(lp.DiseaseState_ABM(sim))
sim.add_component(lp.Transmission_ABM(sim))
# sim.add_component(lp.Migration_ABM(sim))
sim.add_component(lp.VitalDynamics_ABM(sim))
sim.add_component(lp.RI_ABM(sim))
sim.add_component(lp.SIA_ABM(sim))

# Run the simulation
sim.run()

# Plot results
sim.plot()

print("Done.")
