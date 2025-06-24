from pathlib import Path

import numpy as np
import yaml

import laser_polio as lp


def objective(
    trial, calib_config, model_config_path, fit_function, results_path, actual_data_file, n_replicates=1, scoring_fn=None, target_fn=None
):
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

    # Run the simulation n_replicates times
    # Load base config
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)
    # Merge with precedence to Optuna params
    config = {**model_config, **suggested_params}
    config["results_path"] = results_path
    fit_scores = []
    seeds = []
    predictions = []
    for rep in range(n_replicates):
        try:
            # Run sim
            sim = lp.run_sim(config, verbose=0)

            # Record seed (first rep only)
            if rep == 0:
                trial.set_user_attr("rand_seed", sim.pars.seed)

            # Evaluate fit
            actual = target_fn(actual_data_file, model_config_path, is_actual_data=True)
            predicted = target_fn(results_file, model_config_path, is_actual_data=False)
            weights = calib_config.get("metadata", {}).get("weights", {})
            scores = scoring_fn(actual, predicted, weights=weights)
            score = scores["total_log_likelihood"]
            fit_scores.append(score)
            seeds.append(sim.pars.seed)
            predictions.append(predicted)

        except Exception as e:
            print(f"[ERROR] Simulation failed in replicate {rep}: {e}")
            fit_scores.append(float("inf"))

    # Save per-replicate scores & seeds to Optuna
    trial.set_user_attr("actual", json_friendly(actual))
    trial.set_user_attr("predicted", [json_friendly(p) for p in predictions])
    trial.set_user_attr("likelihoods", json_friendly(scores))
    trial.set_user_attr("n_reps", n_replicates)
    trial.set_user_attr("rep_scores", fit_scores)
    trial.set_user_attr("rand_seed", seeds)

    # Return average score
    return np.mean(fit_scores)


def json_friendly(d):
    """Convert a dict of arrays to plain Python types that are JSON serializable."""
    if isinstance(d, dict):
        result = {}
        for k, v in d.items():
            # Convert tuple keys to strings
            if isinstance(k, tuple):
                key_str = str(k)
            else:
                key_str = k

            # Handle nested dictionaries
            if isinstance(v, dict):
                result[key_str] = json_friendly(v)
            # Handle numpy arrays
            elif isinstance(v, np.ndarray):
                result[key_str] = v.tolist()
            # Handle lists (recursively process them)
            elif isinstance(v, list):
                result[key_str] = [json_friendly(item) if isinstance(item, (dict, np.ndarray)) else item for item in v]
            # Handle other types
            else:
                result[key_str] = v
        return result
    elif isinstance(d, np.ndarray):
        return d.tolist()
    else:
        return d
