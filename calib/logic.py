import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import calib_db
import numpy as np
import optuna
import pandas as pd
import yaml

# from logic import objective
import laser_polio as lp


def process_data(filename):
    """Load simulation results and extract features for comparison."""
    df = pd.read_csv(filename)
    return {
        "total_infected": df["I"].sum(),
        "peak_infection_time": df.loc[df["I"].idxmax(), "Time"],
    }


def compute_fit(actual, predicted, use_squared=False, normalize=False, weights=None):
    """Compute distance between actual and predicted summary metrics."""
    fit = 0
    weights = weights or {}

    for key in actual:
        v1 = np.array(actual[key], dtype=float)
        v2 = np.array(predicted[key], dtype=float)
        gofs = np.abs(v1 - v2)

        if normalize and v1.max() > 0:
            gofs /= v1.max()
        if use_squared:
            gofs **= 2

        loss_weight = weights.get(key, 1)
        fit += (gofs * loss_weight).sum()

    return fit


def objective(trial, calib_config, model_config_path, sim_path, results_path, params_file, actual_data_file):
    """Optuna objective function that runs the simulation and evaluates the fit."""
    results_file = results_path / "simulation_results.csv"
    if Path(results_file).exists():
        try:
            Path(results_file).unlink()
        except PermissionError as e:
            print(f"[WARN] Cannot delete file: {e}")

    # Generate suggested parameters from calibration config
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

    # Save parameters to file (used by setup_sim)
    with open(params_file, "w") as f:
        json.dump(suggested_params, f, indent=4)

    # Run simulation using subprocess
    try:
        subprocess.run(
            [
                sys.executable,
                str(sim_path),
                "--model-config",
                str(model_config_path),
                "--params-file",
                str(params_file),
                "--results-path",
                str(results_path),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        return float("inf")

    # Load results and compute fit
    actual = process_data(actual_data_file)
    predicted = process_data(results_file)
    return compute_fit(actual, predicted)


def run_worker_main(
    study_name=None,
    num_trials=None,
    calib_config=None,
    model_config=None,
    results_path=None,
    sim_path=None,
    params_file="params.json",
    actual_data_file=None,
):
    """Run Optuna trials to calibrate the model via CLI or programmatically."""

    # 👇 Provide defaults for programmatic use
    study_name = study_name or "calib_demo_zamfara_r0"
    num_trials = num_trials or 5
    calib_config = calib_config or lp.root / "calib/calib_configs/calib_pars_r0.yaml"
    model_config = model_config or lp.root / "calib/model_configs/config_zamfara.yaml"
    results_path = results_path or lp.root / "calib/results" / study_name
    sim_path = sim_path or lp.root / "calib/setup_sim_v2.py"
    actual_data_file = actual_data_file or lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"

    print(f"[INFO] Running study: {study_name} with {num_trials} trials")
    storage_url = calib_db.get_storage()

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception:
        print(f"[INFO] Creating new study: '{study_name}'")
        study = optuna.create_study(study_name=study_name, storage=storage_url)

    with open(calib_config) as f:
        calib_config_dict = yaml.safe_load(f)

    study.set_user_attr("parameter_spec", calib_config_dict.get("parameters", {}))
    for k, v in calib_config_dict.get("metadata", {}).items():
        study.set_user_attr(k, v)

    wrapped_objective = partial(
        objective,
        calib_config=calib_config_dict,
        model_config_path=Path(model_config),
        sim_path=Path(sim_path),
        results_path=Path(results_path),
        params_file=params_file,
        actual_data_file=Path(actual_data_file),
    )

    study.optimize(wrapped_objective, n_trials=num_trials)

    best = study.best_trial
    print("\nBest Trial:")
    print(f"  Value: {best.value}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    output_dir = Path(results_path)
    output_dir.mkdir(exist_ok=True)

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(output_dir / "calibration_results.csv", index=False)

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=4)
    with open(output_dir / "study_metadata.json", "w") as f:
        json.dump(study.user_attrs, f, indent=4)

    print("✅ Calibration complete. Results saved.")
