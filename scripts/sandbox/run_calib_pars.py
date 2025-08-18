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
model_config_path = Path("calib/model_configs/config_nigeria_7y_2017_region_groupings.yaml")

# Calib pars
suggested_params = {
    "r0": 10.09477395131999,
    "seasonal_amplitude": 0.39921439427854394,
    "seasonal_peak_doy": 217,
    "radiation_k": 2.990077796097201,
}

# Runtime parameters
verbose = 1
results_path = "results/calib_nigeria_7y_2017_underwt_region_groupings_20250814"

######### END OF USER PARS ########
###################################

# Merge parameters into config
with open(model_config_path) as f:
    model_config = yaml.safe_load(f)
config = {**model_config, **suggested_params}
config["results_path"] = results_path
config["save_plots"] = True
config["plot_pars"] = True

# Run simulation
sc.printcyan("🚀 Running simulation...")
sim = lp.run_sim(config, verbose=verbose)

# Print elapsed time
elapsed_time = sc.toc(start_time, output=True)
sc.printcyan("✅ Simulation completed successfully!")
sc.printcyan(f"🕒 Time elapsed: {elapsed_time}")
sc.printcyan(f"💾 Results saved to: {results_path}")
