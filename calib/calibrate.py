import os
import shutil
from pathlib import Path

import calib_db
import click
import optuna
from calib_report import plot_stuff
from calib_report import save_study_results
from logic import run_worker_main

import laser_polio as lp

CONTEXT_SETTINGS = {"help_option_names": ["--help"], "terminal_width": 240}

if os.getenv("POLIO_ROOT"):
    lp.root = Path(os.getenv("POLIO_ROOT"))

# ------------------- USER CONFIG -------------------
num_trials = 2
study_name = "calib_nigeria_smpop_r0_k_seasonality_20250421_date_fix"
calib_config_path = lp.root / "calib/calib_configs/r0_k_seasonality.yaml"
model_config_path = lp.root / "calib/model_configs/config_nigeria_popscale0.01.yaml"
fit_function = "log_likelihood"  # options are "log_likelihood" or "mse"
results_path = lp.root / "results" / study_name
actual_data_file = lp.root / "results" / study_name / "actual_data.csv"
# ---------------------------------------------------


def main(model_config, results_path, study_name, fit_function="mse", **kwargs):
    # Run calibration
    run_worker_main(study_name=study_name, model_config=model_config, results_path=results_path, fit_function=fit_function, **kwargs)

    # Save & plot the calibration results
    shutil.copy(model_config, Path(results_path) / "model_config.yaml")
    storage_url = calib_db.get_storage()
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.results_path = results_path
    study.storage_url = storage_url
    save_study_results(study, Path(results_path))
    if not os.getenv("HEADLESS"):
        plot_stuff(study_name, storage_url, output_dir=Path(results_path))

    print("✅ Calibration complete. Results saved.")


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--study-name", default=study_name, show_default=True)
@click.option("--num-trials", default=num_trials, show_default=True, type=int)
@click.option("--calib-config", default=str(calib_config_path), show_default=True)
@click.option("--model-config", default=str(model_config_path), show_default=True)
@click.option("--fit-function", default=fit_function, show_default=True)
@click.option("--results-path", default=str(results_path), show_default=True)
@click.option("--actual-data-file", default=str(actual_data_file), show_default=True)
def cli(**kwargs):
    main(**kwargs)


if __name__ == "__main__":
    cli()
