# calib/main_calib.py

import json
import subprocess
from functools import partial
from pathlib import Path

import calib_db
import click
import numpy as np
import optuna
import yaml
from calib.archive.logic import compute_fit  # <-- User-configurable logic
from calib.archive.logic import process_data  # <-- User-configurable logic

import laser_polio as lp

# ------------------- USER CONFIG -------------------
study_name = "calib_demo_zamfara_r0"
calib_config = lp.root / "calib/calib_configs/calib_pars_r0.yaml"
model_config = lp.root / "calib/model_configs/config_zamfara.yaml"
setup_sim_path = lp.root / "setup_sim.py"

PARAMS_FILE = "params.json"
RESULTS_FILE = lp.root / "calib/results/calib_demo_zamfara/simulation_results.csv"
ACTUAL_DATA_FILE = lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"

# ---------------------------------------------------


def objective(trial, calib_config, model_config_path):
    """Optuna objective: run model with trial parameters and score result."""
    Path(RESULTS_FILE).unlink(missing_ok=True)

    suggested_params = {}
    for name, spec in calib_config["parameters"].items():
        low = spec["low"]
        high = spec["high"]

        if isinstance(low, int) and isinstance(high, int):
            suggested_params[name] = trial.suggest_int(name, low, high)
        elif isinstance(low, float) or isinstance(high, float):
            suggested_params[name] = trial.suggest_float(name, float(low), float(high))
        else:
            raise TypeError(f"Cannot infer parameter type for '{name}'")

    Path(PARAMS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_FILE, "w") as f:
        json.dump(suggested_params, f, indent=4)

    scores = []
    for _ in range(1):  # Replicates if needed
        try:
            subprocess.run(
                ["python", str(setup_sim_path), "--model-config", str(model_config_path), "--params-file", str(PARAMS_FILE)], check=True
            )
            # subprocess.run(setup_sim(config=model_config.get("setup_sim", {})), check=True)
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed: {e}")
            return float("inf")

        # Load results and compute fit
        actual = process_data(ACTUAL_DATA_FILE)
        predicted = process_data(RESULTS_FILE)
        return compute_fit(actual, predicted)

    Path(RESULTS_FILE).unlink(missing_ok=True)
    return np.mean(scores)


@click.command()
@click.option("--study_name", default=str(study_name), help="Name of the Optuna study.")
@click.option("--n-trials", default=1, type=int, help="Number of optimization trials.")
@click.option("--calib-config", default=str(calib_config), type=str, help="Calibration configuration file.")
@click.option("--model-config", default=str(model_config), type=str, help="Model configuration file.")
def run_worker(study_name, n_trials, calib_config, model_config):
    """Run Optuna trials with imported configuration and scoring logic."""

    storage_url = calib_db.get_storage()
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception:
        print(f"Study '{study_name}' not found. Creating a new study.")
        study = optuna.create_study(study_name=study_name, storage=storage_url)

    with open(calib_config) as f:
        calib_config_dict = yaml.safe_load(f)

    model_config_path = Path(model_config)

    output_dir = Path(study_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    study.set_user_attr("parameter_spec", calib_config_dict.get("parameters", {}))
    for key, value in calib_config_dict.get("metadata", {}).items():
        study.set_user_attr(key, value)

    wrapped_objective = partial(objective, calib_config=calib_config_dict, model_config=model_config_path)
    study.optimize(wrapped_objective, n_trials=n_trials)

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
