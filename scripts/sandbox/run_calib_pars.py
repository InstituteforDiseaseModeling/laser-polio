from pathlib import Path

import sciris as sc
import yaml

import laser_polio as lp

# Record start time
start_time = sc.tic()

"""
This script runs a simulation using a similar configuration to the calibration approach in objective.py. 
It takes as input a model_config.yaml file & a dictionary of parameters. The latter is 
typically printed during calibration runs.
"""

###################################
######### USER PARAMETERS #########

# Load model configuration from YAML file
model_config_path = Path("calib/model_configs/config_west_africa_7y_2017_region_strategy.yaml")

# Calib pars
suggested_params = {
    "r0": 8.498575531913849,
    "seasonal_amplitude": 0.27275401111572795,
    "seasonal_peak_doy": 246,
    "gravity_k_exponent": -15.647094114805695,
    "gravity_a": 1.7518681341407671,
    "gravity_b": 1.0767329166378972,
    "gravity_c": 1.13535137336246,
    "node_seeding_zero_inflation": 0.958063878242425,
    "r0_scalar_wt_slope": 53.9136355397035,
    "r0_scalar_wt_intercept": 0.07092103447398954,
    "r0_scalar_wt_center": 0.9974549294103241,
    "sia_re_center": 0.16373623408279606,
    "sia_re_scale": 0.4613139358510542,
    "init_immun_scalar": 1.064316662988012,
}

# Runtime parameters
verbose = 1
results_path = "results/west_africa_calib_20250807"

######### END OF USER PARS ########
###################################

# Merge parameters into config
with open(model_config_path) as f:
    model_config = yaml.safe_load(f)
config = {**model_config, **suggested_params}
config["results_path"] = results_path

# Run simulation
sc.printcyan("🚀 Running simulation...")
sim = lp.run_sim(config, verbose=verbose)

# Print elapsed time
elapsed_time = sc.toc(start_time, output=True)
sc.printcyan("✅ Simulation completed successfully!")
sc.printcyan(f"🕒 Time elapsed: {elapsed_time}")
sc.printcyan(f"💾 Results saved to: {results_path}")
