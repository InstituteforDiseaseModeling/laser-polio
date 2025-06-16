import os
import shutil
import traceback
from pathlib import Path

import calib_db
import click
import optuna
import sciris as sc
from report import plot_likelihoods
from report import plot_optuna
from report import plot_targets
from report import save_study_results
from worker import run_worker_main

import laser_polio as lp

# ------------------- USER CONFIGS -------------------

# study_name = "calib_config_kano_jigawa_ipv_20250612"
# model_config = "config_kano_jigawa_ipv.yaml"
# calib_config = "r0_k_ssn.yaml"
# job_name = "laser-polio-worker-sk"

study_name = "calib_nigeria_3y_2018_pim_gravity_zinb_ipv_20250613"
model_config = "config_nigeria_3y_2018_pim_gravity_zinb_ipv.yaml"
calib_config = "r0_k_ssn_gravitykabc_zinb.yaml"

fit_function = "log_likelihood"
n_trials = 2
n_replicates = 1  # Number of replicates to run for each trial

# ------------------- END USER CONFIGS -------------------


CONTEXT_SETTINGS = {"help_option_names": ["--help"], "terminal_width": 240}

if os.getenv("POLIO_ROOT"):
    lp.root = Path(os.getenv("POLIO_ROOT"))


def resolve_paths(study_name, model_config, calib_config, results_path=None, actual_data_file=None):
    """
    Build composite paths
    """
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


def main(study_name, model_config, calib_config, fit_function, n_replicates, n_trials, results_path, actual_data_file, dry_run):
    model_config, calib_config, results_path, actual_data_file = resolve_paths(
        study_name, model_config, calib_config, results_path, actual_data_file
    )

    print(f"🔍 Running calibration for study '{study_name}'...")

    Path(results_path).mkdir(parents=True, exist_ok=True)
    # Run calibration and postprocess
    run_worker_main(
        study_name=study_name,
        model_config=model_config,
        calib_config=calib_config,
        fit_function=fit_function,
        n_replicates=n_replicates,
        n_trials=n_trials,
        results_path=results_path,
        actual_data_file=actual_data_file,
        dry_run=dry_run,
    )
    if dry_run:
        return

    shutil.copy(model_config, results_path / "model_config.yaml")

    print("💾 Saving study results...")
    storage_url = calib_db.get_storage()
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.results_path = results_path
    study.storage_url = storage_url
    save_study_results(study, results_path)

    print("📊 Plotting study results...")
    if not os.getenv("HEADLESS"):
        plot_optuna(study_name, storage_url, output_dir=results_path)
        plot_targets(study, output_dir=results_path)
        plot_likelihoods(study, output_dir=results_path, use_log=True)

    sc.printcyan("✅ Calibration complete. Results saved.")


@click.command(context_settings=CONTEXT_SETTINGS)
# The default values used here are from the USER CONFIGS section at the top
@click.option("--study-name", default=study_name, show_default=True)
@click.option("--model-config", default=model_config, show_default=True)
@click.option("--calib-config", default=calib_config, show_default=True)
@click.option("--fit-function", default=fit_function, show_default=True)
@click.option("--n-replicates", default=n_replicates, show_default=True, type=int)
@click.option("--n-trials", default=n_trials, show_default=True, type=int)
@click.option("--dry-run", default=False, show_default=True, type=bool)
def cli(**kwargs):
    # 2 params have None to trigger default behavior. None is not real value.
    main(results_path=None, actual_data_file=None, **kwargs)


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        traceback.print_exc()
        print(f"\n❌ Calibration failed with error: {e}")
        exit(1)
