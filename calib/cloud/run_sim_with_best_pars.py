import json
from pathlib import Path

import sciris as sc
import yaml

import laser_polio as lp

# Record start time
start_time = sc.tic()

"""
This script runs a simulation using the best pars f similar configuration to the calibration approach in objective.py. 
It takes as input a model_config.yaml file & a dictionary of parameters. The latter is 
typically printed during calibration runs.
"""

###################################
######### USER PARAMETERS #########

# Study name
study_name = "calib_nigeria_7y_2017_underwt_regions_maxmigrfrac_dm_fix_20250822"


######### END OF USER PARS ########
###################################

verbose = 1
results_path = Path(f"results/{study_name}")  # Path to the calib results
output_dir = results_path / "best_sim"  # Path to the output directory
best_pars_path = results_path / "best_params.json"  # Path to the best pars for the study
model_config_path = results_path / "model_config.yaml"  # Path to the model config for the study

# Merge parameters into config
with open(model_config_path) as f:
    model_config = yaml.safe_load(f)
with open(best_pars_path) as f:
    best_pars = json.load(f)

config = {**model_config, **best_pars}
config["results_path"] = output_dir
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
