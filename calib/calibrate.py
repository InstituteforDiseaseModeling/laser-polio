import os
import shutil
from pathlib import Path

import calib_db
import click
import optuna
import sciris as sc
from report import plot_likelihoods
from report import plot_stuff
from report import plot_targets
from report import save_study_results
from worker import run_worker_main

import laser_polio as lp

CONTEXT_SETTINGS = {"help_option_names": ["--help"], "terminal_width": 240}

if os.getenv("POLIO_ROOT"):
    lp.root = Path(os.getenv("POLIO_ROOT"))

# ------------------- USER CONFIGS -------------------

study_name = "calib_nigeria_6y_pim_gravity_zinb_birth_fix_20250528"
model_config = "config_nigeria_6y_pim_gravity_zinb.yaml"
calib_config = "r0_k_ssn_gravity_zinb.yaml"
fit_function = "log_likelihood"
num_trials = 2
n_replicates = 1  # Number of replicates to run for each trial

# ---------------------------------------------------

# Set up paths
model_config_path = lp.root / "calib/model_configs" / model_config
calib_config_path = lp.root / "calib/calib_configs" / calib_config
results_path = lp.root / "results" / study_name
actual_data_file = lp.root / "results" / study_name / "actual_data.csv"


def main(model_config, results_path, study_name, fit_function="mse", **kwargs):
    # Resolve paths if needed
    model_config = Path(model_config)
    if not model_config.is_absolute():
        model_config = lp.root / "calib/model_configs" / model_config

    calib_config = Path(kwargs.get("calib_config"))
    if not calib_config.is_absolute():
        calib_config = lp.root / "calib/calib_configs" / calib_config
    kwargs["calib_config"] = calib_config  # update the reference

    results_path = Path(results_path)
    if not results_path.is_absolute():
        results_path = lp.root / "results" / study_name

    actual_data_file = kwargs.get("actual_data_file", results_path / "actual_data.csv")
    if isinstance(actual_data_file, str) or isinstance(actual_data_file, Path):
        actual_data_file = Path(actual_data_file)
    if not actual_data_file.is_absolute():
        actual_data_file = results_path / actual_data_file
    kwargs["actual_data_file"] = actual_data_file

    # Run calibration
    run_worker_main(study_name=study_name, model_config=model_config, results_path=results_path, fit_function=fit_function, **kwargs)

    # Save & plot the calibration results
    Path(results_path).mkdir(parents=True, exist_ok=True)
    shutil.copy(model_config, Path(results_path) / "model_config.yaml")
    storage_url = calib_db.get_storage()
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.results_path = results_path
    study.storage_url = storage_url
    save_study_results(study, Path(results_path))
    if not os.getenv("HEADLESS"):
        plot_stuff(study_name, storage_url, output_dir=Path(results_path))
        plot_targets(study, output_dir=Path(results_path))
        plot_likelihoods(study, output_dir=Path(results_path), use_log=True)

    sc.printcyan("✅ Calibration complete. Results saved.")


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--study-name", default=study_name, show_default=True)
@click.option("--num-trials", default=num_trials, show_default=True, type=int)
@click.option("--calib-config", default=str(calib_config_path), show_default=True)
@click.option("--model-config", default=str(model_config_path), show_default=True)
@click.option("--fit-function", default=fit_function, show_default=True)
@click.option("--results-path", default=str(results_path), show_default=True)
@click.option("--actual-data-file", default=str(actual_data_file), show_default=True)
@click.option("--n-replicates", default=n_replicates, show_default=True, type=int)
def cli(**kwargs):
    main(**kwargs)


if __name__ == "__main__":
    cli()
