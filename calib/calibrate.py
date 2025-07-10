import os
import sys
import shutil
import traceback
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from calib import calib_db
import click
import optuna
import sciris as sc
from calib.report import plot_likelihoods
from calib.report import plot_optuna
from calib.report import plot_targets
from calib.report import save_study_results
from calib.worker import run_worker_main

import laser_polio as lp

# ------------------- USER CONFIGS -------------------

# study_name = "calib_kano_jigawa_ipv_6y_20250619_v2"
# model_config = "config_kano_jigawa_ipv.yaml"
# calib_config = "r0_k_ssn_period.yaml"

study_name = "calib_nigeria_6y_2018_underwt_gravity_zinb_ipv_wts_vxtranszero_20250703"
model_config = "config_nigeria_6y_2018_underwt_gravity_zinb_ipv_moreseeds_alt_zerovxtrans.yaml"
calib_config = "r0_ssn_gravkabc_zinb_r0sclrs_siasclrs_initimmunsclrs_dirichlet_wts.yaml"

fit_function = "log_likelihood"
n_trials = 2
n_replicates = 1  # Number of replicates to run for each trial

# ------------------- END USER CONFIGS -------------------


CONTEXT_SETTINGS = {"help_option_names": ["--help"], "terminal_width": 240}

if os.getenv("POLIO_ROOT"):
    lp.root = Path(os.getenv("POLIO_ROOT"))


def resolve_paths(study_name, model_config, calib_config, results_path, actual_data_file):
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

    #print( f"1. {results_path=}" )
    results_path = Path(results_path) if results_path else root / "results" / study_name
    #print( f"2. {results_path=}" )
    #if not results_path.is_absolute():
    #    results_path = root / "results" / study_name
    #print( f"3. {results_path=}" )

    actual_data_file = Path(actual_data_file) if actual_data_file else results_path / "actual_data.csv"
    #if not actual_data_file.is_absolute():
        #actual_data_file = results_path / actual_data_file

    return model_config, calib_config, results_path, actual_data_file


def main(study_name, model_config, calib_config, fit_function, n_replicates, n_trials, results_path, actual_data_file, dry_run):
    print( f"1. {actual_data_file=}" )
    model_config, calib_config, results_path, actual_data_file = resolve_paths(
        study_name, model_config, calib_config, results_path, actual_data_file
    )
    print( f"2. {actual_data_file=}" )

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
@click.option("--results-path", default=".", show_default=True)
@click.option("--actual-data-file", default=None, show_default=True)
@click.option("--dry-run", default=False, show_default=True, type=bool)
def cli(**kwargs):
    # 2 params have None to trigger default behavior. None is not real value.
    main(**kwargs)


if __name__ == "__main__":
    try:
        cli()
    except Exception as e:
        traceback.print_exc()
        print(f"\n❌ Calibration failed with error: {e}")
        exit(1)
