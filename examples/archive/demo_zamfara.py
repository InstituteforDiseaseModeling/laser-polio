import numpy as np
from laser_core.propertyset import PropertySet

# from laser_polio.utils import *
import laser_polio as lp

###################################
######### USER PARAMETERS #########

regions = ["ZAMFARA"]
start_date = lp.date("2025-01-01")
n_days = 180
pop_scale = 1 / 10

######### END OF USER PARS ########
###################################


# Load region data: population, birth rates (CBR), distance matrix, & vaccination schedule
dot_names = lp.find_matching_dot_names(regions, "data/sia_scenario_1.csv")  # TODO: Replace this with a built to purpose reference file
pop = np.random.randint(100000, 500000, len(dot_names))  # TODO: Replace this with actual population data
pop = pop * pop_scale  # Scale down population for speed
cbr = np.full(len(dot_names), 44.0)  # TODO: Replace this with actual data
dist_matrix = lp.get_distance_matrix("data/distance_matrix_africa_adm2.h5", dot_names)  # Load distances matrix (km)
sia_schedule = lp.process_sia_schedule_polio("data/sia_scenario_1.csv", dot_names, start_date)  # Load sia schedule

# Setup placeholder pars
init_prev = np.zeros(len(pop))  # TODO: Replace with estimates from regression model?
init_prev[0] = 0.01  # 1% infected in the first region
beta_spatial = np.random.uniform(0.5, 2, len(pop))  # TODO: Replace with some sort of spatial transmission scalar
vx_prob_ri = np.random.uniform(0, 0.5, len(pop))  # TODO: Replace with actual data
vx_prob_sia = np.random.uniform(0.5, 0.9, len(pop))  # TODO: Replace with actual data

# Set parameters
pars = PropertySet(
    {
        # Time
        "start_date": start_date,  # Start date of the simulation
        "dur": n_days,  # Number of timesteps
        # Population
        "n_ppl": pop,  # np.array([30000, 10000, 15000, 20000, 25000]),
        "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",  # From https://www.populationpyramid.net/nigeria/2024/
        "cbr": cbr,  # np.array([37, 41, 30, 25, 33]),  # Crude birth rate per 1000 per year
        # Disease
        "init_prev": init_prev,  # Initial prevalence per node (1% infected)
        "r0": 14,  # Basic reproduction number
        "risk_mult_var": 4.0,  # Lognormal variance for the individual-level risk multiplier (risk of acquisition multiplier; mean = 1.0)
        "corr_risk_inf": 0.8,  # Correlation between individual risk multiplier and individual infectivity (daily infectivity, mean = 14/24)
        "beta_spatial": beta_spatial,  # Spatial transmission scalar (multiplied by global rate)
        "seasonal_factor": 0.125,  # Seasonal variation in transmission
        "seasonal_phase": 180,  # Phase of seasonal variation
        "p_paralysis": 1 / 20,  # Probability of paralysis
        "dur_exp": lp.normal(mean=3, std=1),  # Duration of the exposed state
        "dur_inf": lp.gamma(shape=4.51, scale=5.32),  # Duration of the infectious state
        # Migration
        "distances": dist_matrix,  # Distance in km between nodes
        "gravity_k": 0.5,  # Gravity scaling constant
        "gravity_a": 1,  # Origin population exponent
        "gravity_b": 1,  # Destination population exponent
        "gravity_c": 2.0,  # Distance exponent
        "max_migr_frac": 0.01,  # Fraction of population that migrates
        # Interventions
        "vx_prob_ri": vx_prob_ri,  # Probability of routine vaccination
        "sia_schedule": sia_schedule,  # Schedule of SIAs
        "vx_prob_sia": vx_prob_sia,  # Effectiveness of SIAs
    }
)

# Initialize the sim
sim = lp.SEIR_ABM(pars)
sim.add_component(lp.DiseaseState_ABM(sim))
sim.add_component(lp.Transmission_ABM(sim))
sim.add_component(lp.VitalDynamics_ABM(sim))
sim.add_component(lp.RI_ABM(sim))
sim.add_component(lp.SIA_ABM(sim))

# Run the simulation
sim.run()

# Plot results
sim.plot()

print("Done.")
