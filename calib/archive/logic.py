# from functools import partial
# from pathlib import Path

# import calib_db
# import optuna
# import yaml

# # from logic import objective
# import laser_polio as lp

# # def calc_calib_targets_paralysis(filename, model_config_path=None, is_actual_data=True):
# #     """Load simulation results and extract features for comparison."""

# #     # Load the data & config
# #     df = pd.read_csv(filename)
# #     with open(model_config_path) as f:
# #         model_config = yaml.safe_load(f)

# #     # Parse dates to datetime object if needed
# #     if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
# #         df["date"] = pd.to_datetime(df["date"])
# #     df["year"] = df["date"].dt.year
# #     df["month"] = df["date"].dt.month

# #     # Choose the column to summarize
# #     if is_actual_data:
# #         case_col = "P"
# #         scale_factor = 1.0
# #     else:
# #         case_col = "new_exposed"
# #         scale_factor = 1 / 2000.0
# #         # The actual data is in months & the sim has a tendency to rap into the next year (e.g., 2020-01-01) so we need to exclude and dates beyond the last month of the actual data
# #         max_date = lp.find_latest_end_of_month(df["date"])
# #         df = df[df["date"] <= max_date]

# #     targets = {}

# #     # 1. Total infected (scaled if simulated)
# #     targets["total_infected"] = np.array(df[case_col].sum() * scale_factor)

# #     # 2. Yearly cases
# #     targets["yearly_cases"] = df.groupby("year")[case_col].sum().values * scale_factor

# #     # 3. Monthly cases
# #     targets["monthly_cases"] = df.groupby("month")[case_col].sum().values * scale_factor

# #     # 4. Regional group cases
# #     if model_config and "summary_config" in model_config:
# #         region_groups = model_config["summary_config"].get("region_groups", {})
# #         regional_cases = []
# #         for name in region_groups:
# #             node_list = region_groups[name]
# #             total = df[df["node"].isin(node_list)][case_col].sum() * scale_factor
# #             regional_cases.append(total)
# #         targets["regional_cases"] = np.array(regional_cases)

# #     print(f"{targets=}")
# #     return targets


# # def calc_calib_targets(filename, model_config_path=None):
# #     """Load simulation results and extract features for comparison."""

# #     # Load the data & config
# #     df = pd.read_csv(filename)
# #     with open(model_config_path) as f:
# #         model_config = yaml.safe_load(f)

# #     # Parse dates to datetime object if needed
# #     if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
# #         df["date"] = pd.to_datetime(df["date"])
# #     df["month"] = df["date"].dt.month

# #     targets = {}

# #     # 1. Total infected
# #     targets["total_infected"] = df["I"].sum()

# #     # 2. Yearly cases

# #     # 3. Monthly cases
# #     targets["monthly_cases"] = df.groupby("month")["I"].sum().values

# #     # 4. Regional group cases as a single array
# #     if model_config and "summary_config" in model_config:
# #         region_groups = model_config["summary_config"].get("region_groups", {})
# #         regional_cases = []
# #         for name in region_groups:
# #             node_list = region_groups[name]
# #             total = df[df["node"].isin(node_list)]["I"].sum()
# #             regional_cases.append(total)
# #         targets["regional_cases"] = np.array(regional_cases)

# #     print(f"{targets=}")
# #     return targets


# # def process_data(filename):
# #     """Load simulation results and extract features for comparison."""
# #     df = pd.read_csv(filename)
# #     return {
# #         "total_infected": df["I"].sum(),
# #         "peak_infection_time": df.loc[df["I"].idxmax(), "Time"],
# #     }


# # def compute_fit(actual, predicted, use_squared=False, normalize=False, weights=None):
# #     """Compute distance between actual and predicted summary metrics."""
# #     fit = 0
# #     weights = weights or {}

# #     for key in actual:
# #         if key not in predicted:
# #             print(f"[WARN] Key missing in predicted: {key}")
# #             continue

# #         try:
# #             v1 = np.array(actual[key], dtype=float)
# #             v2 = np.array(predicted[key], dtype=float)

# #             if v1.shape != v2.shape:
# #                 sc.printyellow(f"[WARN] Shape mismatch on '{key}': {v1.shape} vs {v2.shape}")
# #                 continue

# #             gofs = np.abs(v1 - v2)

# #             if normalize and v1.max() > 0:
# #                 gofs = gofs / v1.max()
# #             if use_squared:
# #                 gofs = gofs**2

# #             weight = weights.get(key, 1)
# #             fit += (gofs * weight).sum()

# #         except Exception as e:
# #             print(f"[ERROR] Skipping '{key}' due to: {e}")

# #     return fit


# # def compute_log_likelihood_fit(actual, predicted, method="poisson", dispersion=1.0, weights=None):
# #     """
# #     Compute log-likelihood of actual data given predicted data.

# #     Parameters:
# #         actual (dict): Dict of observed summary statistics.
# #         predicted (dict): Dict of simulated summary statistics.
# #         method (str): Distribution to use ("poisson" or "neg_binomial").
# #         dispersion (float): Dispersion parameter for neg_binomial (var = mu + mu^2 / r).
# #         weights (dict): Optional weights for each target.

# #     Returns:
# #         float: Total log-likelihood (higher is better).
# #     """
# #     log_likelihood = 0.0
# #     weights = weights or {}

# #     for key in actual:
# #         if key not in predicted:
# #             print(f"[WARN] Key missing in predicted: {key}")
# #             continue

# #         try:
# #             v_obs = np.array(actual[key], dtype=float)
# #             v_sim = np.array(predicted[key], dtype=float)
# #             v_sim = np.clip(v_sim, 1e-6, None)  # Prevent log(0) in Poisson

# #             if v_obs.shape != v_sim.shape:
# #                 sc.printyellow(f"[WARN] Shape mismatch on '{key}': {v_obs.shape} vs {v_sim.shape}")
# #                 continue

# #             if method == "poisson":
# #                 logp = poisson.logpmf(v_obs, v_sim)
# #             elif method == "neg_binomial":
# #                 # NB parameterization via mean (mu) and dispersion (r)
# #                 # r = dispersion; p = r / (r + mu)
# #                 mu = v_sim
# #                 r = dispersion
# #                 p = r / (r + mu)
# #                 logp = nbinom.logpmf(v_obs, r, p)
# #             else:
# #                 raise ValueError(f"Unknown method '{method}'")

# #             # Sum log-likelihoods, but normalize by number of observations (e.g., total_infected has 1 value, while monthly_cases has 12)
# #             n = len(logp)
# #             weight = weights.get(key, 1)
# #             log_likelihood += weight * logp.sum() / n

# #         except Exception as e:
# #             print(f"[ERROR] Skipping '{key}' due to: {e}")

# #     return log_likelihood


# # def objective(trial, calib_config, model_config_path, fit_function, results_path, actual_data_file, n_replicates=1):
# #     """Optuna objective function that runs the simulation and evaluates the fit."""
# #     results_file = results_path / "simulation_results.csv"
# #     if Path(results_file).exists():
# #         try:
# #             Path(results_file).unlink()
# #         except PermissionError as e:
# #             print(f"[WARN] Cannot delete file: {e}")

# #     # Generate suggested parameters from calibration config
# #     suggested_params = {}
# #     for name, spec in calib_config["parameters"].items():
# #         low = spec["low"]
# #         high = spec["high"]

# #         if isinstance(low, int) and isinstance(high, int):
# #             suggested_params[name] = trial.suggest_int(name, low, high)
# #         elif isinstance(low, float) or isinstance(high, float):
# #             suggested_params[name] = trial.suggest_float(name, float(low), float(high))
# #         else:
# #             raise TypeError(f"Cannot infer parameter type for '{name}'")

# #     # Run the simulation n_replicates times
# #     # Load base config
# #     with open(model_config_path) as f:
# #         model_config = yaml.safe_load(f)
# #     # Merge with precedence to Optuna params
# #     config = {**model_config, **suggested_params}
# #     config["results_path"] = results_path
# #     fit_scores = []
# #     seeds = []
# #     for rep in range(n_replicates):
# #         try:
# #             # Run sim
# #             sim = lp.run_sim(config, verbose=0)

# #             # Record seed (first rep only)
# #             if rep == 0:
# #                 trial.set_user_attr("rand_seed", sim.pars.seed)

# #             # Evaluate fit
# #             actual = calc_calib_targets_paralysis(actual_data_file, model_config_path, is_actual_data=True)
# #             predicted = calc_calib_targets_paralysis(results_file, model_config_path, is_actual_data=False)
# #             if fit_function == "log_likelihood":
# #                 score = -compute_log_likelihood_fit(actual, predicted, method="poisson")  # NEGATE for Optuna
# #             else:
# #                 score = compute_fit(actual, predicted)

# #             fit_scores.append(score)
# #             seeds.append(sim.pars.seed)

# #         except Exception as e:
# #             print(f"[ERROR] Simulation failed in replicate {rep}: {e}")
# #             fit_scores.append(float("inf"))

# #     # Save per-replicate scores & seeds to Optuna
# #     trial.set_user_attr("replicates", n_replicates)
# #     trial.set_user_attr("replicate_scores", fit_scores)
# #     trial.set_user_attr("rand_seed", seeds)

# #     # Return average score
# #     return np.mean(fit_scores)


# def run_worker_main(
#     study_name=None,
#     num_trials=None,
#     calib_config=None,
#     model_config=None,
#     fit_function=None,
#     results_path=None,
#     actual_data_file=None,
#     n_replicates=None,
# ):
#     """Run Optuna trials to calibrate the model via CLI or programmatically."""

#     # 👇 Provide defaults for programmatic use
#     num_trials = num_trials or 5
#     calib_config = calib_config or lp.root / "calib/calib_configs/calib_pars_r0.yaml"
#     model_config = model_config or lp.root / "calib/model_configs/config_zamfara.yaml"
#     fit_function = fit_function or "mse"  # options are "log_likelihood" or "mse"
#     results_path = results_path or lp.root / "calib/results" / study_name
#     actual_data_file = actual_data_file or lp.root / "examples/calib_demo_zamfara/synthetic_infection_counts_zamfara_250.csv"
#     n_replicates = n_replicates or 1

#     print(f"[INFO] Running study: {study_name} with {num_trials} trials")
#     storage_url = calib_db.get_storage()

#     # sampler = optuna.samplers.RandomSampler(seed=42)  # seed is optional for reproducibility
#     try:
#         study = optuna.load_study(study_name=study_name, storage=storage_url)  # , sampler=sampler)
#     except Exception:
#         print(f"[INFO] Creating new study: '{study_name}'")
#         study = optuna.create_study(study_name=study_name, storage=storage_url)

#     with open(calib_config) as f:
#         calib_config_dict = yaml.safe_load(f)

#     study.set_user_attr("parameter_spec", calib_config_dict.get("parameters", {}))
#     for k, v in calib_config_dict.get("metadata", {}).items():
#         study.set_user_attr(k, v)

#     wrapped_objective = partial(
#         objective,
#         calib_config=calib_config_dict,
#         model_config_path=Path(model_config),
#         fit_function=fit_function,
#         results_path=Path(results_path),
#         actual_data_file=Path(actual_data_file),
#         n_replicates=n_replicates,
#     )

#     study.optimize(wrapped_objective, n_trials=num_trials)
