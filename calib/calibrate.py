import os
from pathlib import Path

import click
from logic import run_worker_main

import laser_polio as lp

CONTEXT_SETTINGS = {"help_option_names": ["--help"], "terminal_width": 240}

if os.getenv("POLIO_ROOT"):
    lp.root = Path(os.getenv("POLIO_ROOT"))

# ------------------- USER CONFIG -------------------
num_trials = 2
study_name = "calib_nigeria_smpop_r0_k_seasonality"
calib_config_path = lp.root / "calib/calib_configs/r0_k_seasonality.yaml"
model_config_path = lp.root / "calib/model_configs/config_nigeria_popscale0.0001.yaml"
sim_path = lp.root / "calib/setup_sim.py"
results_path = lp.root / "calib/results" / study_name
params_file = "params.json"
actual_data_file = lp.root / "calib/results/" / study_name / "actual_data.csv"
# ---------------------------------------------------


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--study-name", default=study_name, show_default=True, help="Name of the Optuna study.")
@click.option("--num-trials", default=num_trials, show_default=True, type=int, help="Number of optimization trials.")
@click.option("--calib-config", default=str(calib_config_path), show_default=True, help="Path to calibration parameter file.")
@click.option("--model-config", default=str(model_config_path), show_default=True, help="Path to model base config.")
@click.option("--results-path", default=str(results_path), show_default=True)
@click.option("--sim-path", default=str(sim_path), show_default=True)
@click.option("--params-file", default=str(params_file), show_default=True)
@click.option("--actual-data-file", default=str(actual_data_file), show_default=True)
def run_worker(**kwargs):
    run_worker_main(**kwargs)


if __name__ == "__main__":
    run_worker()
