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

DEFAULT_STUDY_NAME = "calib_nigeria_6y_pim_gravity_zinb_20250523"
DEFAULT_MODEL_CONFIG = "config_nigeria_6y_pim_gravity_zinb.yaml"
DEFAULT_CALIB_CONFIG = "r0_k_ssn_gravity_zinb.yaml"
DEFAULT_FIT_FUNCTION = "log_likelihood"
DEFAULT_N_REPLICATES = 1


def resolve_paths(study_name, model_config, calib_config, results_path=None, actual_data_file=None):
    root = lp.root

    model_config = Path(model_config)
    if not model_config.is_absolute():
        model_config = root / "calib/model_configs" / model_config

    calib_config = Path(calib_config)
    if not calib_config.is_absolute():
        calib_config = root / "calib/calib_configs" / calib_config

    results_path = Path(results_path) if results_path else root / "results" / study_name
    if not results_path.is_absolute():
        results_path = root / "results" / study_name

    actual_data_file = Path(actual_data_file) if actual_data_file else results_path / "actual_data.csv"
    if not actual_data_file.is_absolute():
        actual_data_file = results_path / actual_data_file

    return model_config, calib_config, results_path, actual_data_file


def main(study_name, model_config, calib_config, results_path=None, actual_data_file=None, fit_function="mse", dry_run=False, **kwargs):
    model_config, calib_config, results_path, actual_data_file = resolve_paths(
        study_name, model_config, calib_config, results_path, actual_data_file
    )
    kwargs.update(
        {
            "calib_config": calib_config,
            "actual_data_file": actual_data_file,
        }
    )

    # Run calibration and postprocess
    run_worker_main(
        study_name=study_name, model_config=model_config, results_path=results_path, fit_function=fit_function, dry_run=dry_run, **kwargs
    )
    if dry_run:
        return

    Path(results_path).mkdir(parents=True, exist_ok=True)
    shutil.copy(model_config, results_path / "model_config.yaml")

    storage_url = calib_db.get_storage()
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.results_path = results_path
    study.storage_url = storage_url

    save_study_results(study, results_path)

    if not os.getenv("HEADLESS"):
        plot_stuff(study_name, storage_url, output_dir=results_path)
        plot_targets(study, output_dir=results_path)
        plot_likelihoods(study, output_dir=results_path, use_log=True)

    sc.printcyan("✅ Calibration complete. Results saved.")


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--study-name", default=DEFAULT_STUDY_NAME, show_default=True)
@click.option("--model-config", default=DEFAULT_MODEL_CONFIG, show_default=True)
@click.option("--calib-config", default=DEFAULT_CALIB_CONFIG, show_default=True)
@click.option("--fit-function", default=DEFAULT_FIT_FUNCTION, show_default=True)
@click.option("--n-replicates", default=DEFAULT_N_REPLICATES, show_default=True, type=int)
@click.option("--num-trials", default=2, show_default=True, type=int)
@click.option("--dry-run", default=False, show_default=True, type=bool)
def cli(**kwargs):
    main(**kwargs)


if __name__ == "__main__":
    cli()
