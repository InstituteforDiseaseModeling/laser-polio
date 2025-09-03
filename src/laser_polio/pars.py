import datetime

import numpy as np
from laser_core.propertyset import PropertySet

import laser_polio as lp

__all__ = ["default_pars", "default_run_order"]

default_pars = PropertySet(
    {
        # Random seed
        "seed": None,
        # Time
        "start_date": datetime.date(2019, 1, 1),  # Start date of the simulation
        "dur": 30,  # Number of timesteps
        # Population
        "init_pop": [15000, 10000],  # Starting population for all nodes. Includes susceptible and recovered
        "init_immun": [
            0.0,
            0.0,
        ],  # Proportion of init_pop that is immune per node. 0.0 = no immunity, 1.0 = full immunity. Expected to be a list of length len(init_pop)
        "init_sus_by_age": None,  # Dataframe containing susceptible counts by age and node. Expected columns: dot_name, node_id, age_min_yr, age_max_yr, n_susceptible
        "age_pyramid_path": "data/Nigeria_age_pyramid_2024.csv",
        "cbr": [37, 41],  # Crude birth rate per 1000 per year
        # Disease
        "strain_ids": {"VDPV2": 0, "Sabin2": 1, "nOPV2": 2},
        "strain_r0_scalars": {
            0: 1.0,  # VDPV2
            1: 0.25,  # Sabin2
            2: 0.125,  # nOPV2
        },
        "init_prev": [0.0, 0.0],  # Initial prevalence per node
        "seed_schedule": None,  # Schedule for seeding cases (list of dicts with either: 1)'date', 'dot_name', 'prevalence' or 2)'timestep', 'node_id', 'prevalence')
        "r0": 14,  # Basic reproduction number
        "r0_scalars": [0.8, 1.2],  # Spatial transmission scalar (multiplied by global rate)
        "seasonal_amplitude": 0.125,  # Seasonal variation in transmission
        "seasonal_peak_doy": 180,  # Phase of seasonal variation
        "individual_heterogeneity": True,  # Whether to use individual heterogeneity in acquisition and infectivity or set to mean values
        "risk_mult_var": 4.0,  # Lognormal variance for the individual-level risk multiplier (risk of acquisition multiplier; mean = 1.0)
        "corr_risk_inf": 0.8,  # Correlation coefficient between individual risk multiplier and individual infectivity (daily infectivity, mean = 14/24)
        "dur_exp": lp.poisson(lam=3),  # Duration of the exposed state
        "dur_inf": lp.gamma(shape=4.51, scale=5.32),  # Duration of the infectious state
        "t_to_paralysis": lp.lognormal(
            mean=12.5, sigma=3.5
        ),  # Time from exposure to paralysis (P) (med ~12 days, mean ~12.5 days, 95% CI: 7-21 days) from Sartwell 1952
        "p_paralysis": 1 / 2000,  # Probability of paralysis
        # Geography
        "shp": None,  # Shapefile of the region
        "node_lookup": None,  # Node info (node_id are keys, dict contains dot_name, lat, lon)
        "distances": np.array([[0, 100], [100, 0]]),  # Distance in km between nodes
        # Migration
        "node_seeding_dispersion": 1000,  # INTEGER (or will round) - negative binomial "k" parameter for the first importation into each node. Larger values -> Poisson.
        "node_seeding_zero_inflation": 0.0,  # Fraction of node seeding events to zero out, float value between 0 and 1; 0.0 -> no zero inflation.
        "migration_method": "radiation",  # Migration method: "gravity" or "radiation"
        "radiation_k_log10": -0.3,  # Radiation model scaling constant in log10 space (10^-0.30103 ≈ 0.5). Based on testing, this par should be between -3.0 and 1.0 for Nigeria.
        "gravity_k": 1.0,  # Gravity scaling constant
        "gravity_k_exponent": 0.0,  # Exponent of the gravity_k term
        "gravity_a": 1,  # Origin population exponent
        "gravity_b": 1,  # Destination population exponent
        "gravity_c": 2.0,  # Distance exponent
        "max_migr_frac": 0.1,  # Max fraction of population that migrates
        # Interventions
        "vx_prob_ri": None,  # Probability of being protected/recovered from RI. Should include coverage and efficacy from expected number of RI doses
        "vx_prob_ipv": None,  # Probability of receiving an IPV dose
        "ipv_start_year": 2015,  # Year to start IPV vaccination
        "sia_schedule": None,
        "vx_prob_sia": None,
        "missed_frac": 0.0,  # Fraction of population that's inaccessible to vaccination
        "vx_efficacy": {
            "perfect": 1.0,
            "bOPV": 0,
            "f-IPV": 0,
            "IPV": 0,
            "IPV + bOPV": 0,
            "mOPV2": 0.7,
            "nOPV2": 0.7 * 0.8,
            "nOPV2 + fIPV": 0.7 * 0.8,
            "topv": 0.5,
        },
        # Component step sizes
        "step_size_VitalDynamics_ABM": 7,
        "step_size_DiseaseState_ABM": 1,
        "step_size_RI_ABM": 14,
        "step_size_SIA_ABM": 1,
        "step_size_Transmission_ABM": 1,
        # Actual data & calibration configs
        "actual_data": None,  # Actual dataset
        "summary_config": None,  # Summary configuration for calibration plotting
        # Verbosity
        "verbose": 1,  # Logging level: 0=silent (warnings only), 1=info, 2=debug, 3=debug+validation
        # Stopping rules
        "stop_if_no_cases": True,  # Stop if no E, I, or seed_schedules remain
    }
)


# Order in which to run model components
default_run_order = ["VitalDynamics_ABM", "DiseaseState_ABM", "RI_ABM", "SIA_ABM", "Transmission_ABM"]
