import datetime

import numpy as np
from laser_core.propertyset import PropertySet

import laser_polio as lp

__all__ = ["default_pars"]

default_pars = PropertySet(
    {
        # Time
        "start_date": datetime.date(2019, 1, 1),  # Start date of the simulation
        "dur": 30,  # Number of timesteps
        # Population
        "n_ppl": [15000, 10000],  # Number of initial agents for each node
        "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",
        "cbr": [37, 41],  # Crude birth rate per 1000 per year
        # Disease
        "init_immun": [0.0, 0.0],  # Initial immunity per node
        "init_prev": [0.0, 0.0],  # Initial prevalence per node
        "r0": 14,  # Basic reproduction number
        "beta_spatial": [0.8, 1.2],  # Spatial transmission scalar (multiplied by global rate)
        "seasonal_factor": 0.125,  # Seasonal variation in transmission
        "seasonal_phase": 180,  # Phase of seasonal variation
        "risk_mult_var": 4.0,  # Lognormal variance for the individual-level risk multiplier (risk of acquisition multiplier; mean = 1.0)
        "corr_risk_inf": 0.8,  # Correlation between individual risk multiplier and individual infectivity (daily infectivity, mean = 14/24)
        "dur_exp": lp.normal(mean=3, std=1),  # Duration of the exposed state
        "dur_inf": lp.gamma(shape=4.51, scale=5.32),  # Duration of the infectious state
        "p_paralysis": 1 / 2000,  # Probability of paralysis
        # Migration
        "distances": [[0, 100], [100, 0]],  # Distance in km between nodes
        "gravity_k": 0.5,  # Gravity scaling constant
        "gravity_a": 1,  # Origin population exponent
        "gravity_b": 1,  # Destination population exponent
        "gravity_c": 2.0,  # Distance exponent
        "max_migr_frac": 0.01,  # Fraction of population that migrates
        "centroids": {
            "CENTER_LAT": np.array([0, 0]),
            "CENTER_LON": np.array([0, 1]),
        },
        # Interventions
        "vx_prob_ri": None,
        "sia_schedule": None,
        "sia_eff": None,
        "life_expectancies": np.ones(5) * 65,  # example: 5 nodes
    }
)
