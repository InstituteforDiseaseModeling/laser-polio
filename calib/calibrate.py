# calib/main_calib.py

import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import calib_db
import click
import numpy as np
import optuna
import yaml
from logic import compute_fit  # <-- User-configurable logic
from logic import process_data  # <-- User-configurable logic

import laser_polio as lp

# ------------------- USER CONFIG -------------------
model_script = Path(lp.root / "calib/demo_zamfara.py").resolve(strict=True)
PARAMS_FILE = "params.json"
RESULTS_FILE = lp.root / "calib/results/calib_demo_zamfara/simulation_results.csv"
ACTUAL_DATA_FILE = lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"
# ---------------------------------------------------


def get_native_runstring():
    return [sys.executable, str(model_script)]


def objective(trial, calib_pars, config_pars):
    """Optuna objective: run model with trial parameters and score result."""
    Path(RESULTS_FILE).unlink(missing_ok=True)

    params = config_pars["parameters"]
    for name, spec in calib_pars["parameters"].items():
        low = spec["low"]
        high = spec["high"]

        if isinstance(low, int) and isinstance(high, int):
            params[name] = trial.suggest_int(name, low, high)
        elif isinstance(low, float) or isinstance(high, float):
            params[name] = trial.suggest_float(name, float(low), float(high))
        else:
            raise TypeError(f"Cannot infer parameter type for '{name}'")

    Path(PARAMS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)

    scores = []
    for _ in range(1):  # Replicates if needed
        try:
            subprocess.run(get_native_runstring(), check=True)
            actual = process_data(ACTUAL_DATA_FILE)
            predicted = process_data(RESULTS_FILE)
            scores.append(compute_fit(actual, predicted))
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed: {e}")
            return float("inf")

    Path(RESULTS_FILE).unlink(missing_ok=True)
    return np.mean(scores)


@click.command()
@click.option("--study-name", default="laser_polio_test", help="Name of the Optuna study.")
@click.option("--num-trials", default=1, type=int, help="Number of optimization trials.")
@click.option("--calib-pars", default="calib_pars.yaml", type=str, help="Calibration parameter file.")
@click.option("--config-pars", default="config.yaml", type=str, help="Base configuration file.")
def run_worker(study_name, num_trials, calib_pars, config_pars):
    """Run Optuna trials with imported configuration and scoring logic."""

    storage_url = calib_db.get_storage()
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception:
        print(f"Study '{study_name}' not found. Creating a new study.")
        study = optuna.create_study(study_name=study_name, storage=storage_url)

    with open(calib_pars) as f:
        calib_pars_dict = yaml.safe_load(f)

    with open(config_pars) as f:
        config_pars_dict = yaml.safe_load(f)

    run_name = calib_pars_dict["metadata"]["name"]
    output_dir = Path(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    study.set_user_attr("parameter_spec", calib_pars_dict.get("parameters", {}))
    for key, value in calib_pars_dict.get("metadata", {}).items():
        study.set_user_attr(key, value)

    wrapped_objective = partial(objective, calib_pars=calib_pars_dict, config_pars=config_pars_dict)
    study.optimize(wrapped_objective, n_trials=num_trials)

    # Output results
    best = study.best_trial
    print("\nBest Trial:")
    print(f"  Value: {best.value}")
    print("  Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    print("\nMetadata:")
    for k, v in study.user_attrs.items():
        print(f"  {k}: {v}")

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(output_dir / "calibration_results.csv", index=False)
    print("✅ Wrote all trial results to calibration_results.csv")

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=4)
    with open(output_dir / "study_metadata.json", "w") as f:
        json.dump(study.user_attrs, f, indent=4)
    print("✅ Saved best parameter set and metadata.")


if __name__ == "__main__":
    run_worker()
