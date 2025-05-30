import numpy as np
import sciris as sc
from scipy.stats import dirichlet_multinomial
from scipy.stats import nbinom
from scipy.stats import poisson


def compute_fit(actual, predicted, use_squared=False, normalize=False, weights=None):
    """Compute distance between actual and predicted summary metrics."""
    fit = 0
    weights = weights or {}

    for key in actual:
        if key not in predicted:
            print(f"[WARN] Key missing in predicted: {key}")
            continue

        try:
            v1 = np.array(actual[key], dtype=float)
            v2 = np.array(predicted[key], dtype=float)

            if v1.shape != v2.shape:
                sc.printyellow(f"[WARN] Shape mismatch on '{key}': {v1.shape} vs {v2.shape}")
                continue

            gofs = np.abs(v1 - v2)

            if normalize and v1.max() > 0:
                gofs = gofs / v1.max()
            if use_squared:
                gofs = gofs**2

            weight = weights.get(key, 1)
            fit += (gofs * weight).sum()

        except Exception as e:
            print(f"[ERROR] Skipping '{key}' due to: {e}")

    return fit


def compute_log_likelihood_fit(actual, predicted, method="poisson", dispersion=1.0, weights=None, norm_by_n=True):
    """
    Compute log-likelihood of actual data given predicted data.

    Parameters:
        actual (dict): Dict of observed summary statistics.
        predicted (dict): Dict of simulated summary statistics.
        method (str): Distribution to use ("poisson" or "neg_binomial").
        dispersion (float): Dispersion parameter for neg_binomial (var = mu + mu^2 / r).
        weights (dict): Optional weights for each target.

    Returns:
        float: Total log-likelihood (higher is better).
    """
    log_likelihoods = {}
    weights = weights or {}

    # Validate weight keys
    actual_keys = set(actual.keys())
    weight_keys = set(weights.keys())
    if not actual_keys <= weight_keys:
        missing = actual_keys - weight_keys
        print(f"[WARN] Missing weights for: {missing}. Defaulting to 1.0.")

    for key in actual:
        try:
            v_obs = np.array(actual[key], dtype=float)
            v_sim = np.array(predicted[key], dtype=float)
            v_sim = np.clip(v_sim, 1e-6, None)  # Prevent log(0) in Poisson

            if v_obs.shape != v_sim.shape:
                sc.printyellow(f"[WARN] Shape mismatch on '{key}': {v_obs.shape} vs {v_sim.shape}")
                continue

            if method == "poisson":
                logp = poisson.logpmf(v_obs, v_sim)
                ll = logp.sum()

            elif method == "neg_binomial":
                # NB parameterization via mean (mu) and dispersion (r)
                # r = dispersion; p = r / (r + mu)
                mu = v_sim
                r = dispersion
                p = r / (r + mu)
                logp = nbinom.logpmf(v_obs, r, p)
                ll = logp.sum()

            elif method == "dirichlet_multinomial":
                # v_obs is alpha for the Dirichlet, v_sim is expected counts
                # We interpret v_obs as prior (alpha = actual + 1), v_sim as test counts
                alpha = v_obs + 1
                n = v_sim.sum()
                x = np.round(v_sim).astype(int)
                ll = dirichlet_multinomial.logpmf(x=x, n=int(n), alpha=alpha)

            else:
                raise ValueError(f"Unknown method '{method}'")

            # Sum log-likelihoods, but normalize by number of observations (e.g., total_infected has 1 value, while monthly_cases has 12)
            weight = weights.get(key, 1)
            n = len(v_obs)
            normalizer = n if norm_by_n and method != "dirichlet_multinomial" else 1
            neg_ll = -1.0 * weight * ll / normalizer  # NEGATE for Optuna
            log_likelihoods[key] = neg_ll

        except Exception as e:
            print(f"[ERROR] Skipping '{key}' due to: {e}")

    total_ll = sum(log_likelihoods.values())
    log_likelihoods["total_log_likelihood"] = total_ll

    return log_likelihoods


def compute_nll(actual, predicted, methods=None, weights=None, dispersion=1.0, norm_by_n=True):
    """
    Compute log-likelihood of actual data given predicted data, with flexible method per key.

    Parameters:
        actual (dict): Observed summary statistics.
        predicted (dict): Simulated summary statistics.
        methods (dict): Mapping from key to method ("poisson", "neg_binomial", "dirichlet_multinomial").
        dispersion (float): Dispersion for neg_binomial or alpha scale for DirMult.
        weights (dict): Optional weights for each key.
        norm_by_n (bool): Whether to normalize likelihood by length.

    Returns:
        dict: Per-key log-likelihood and total.
    """
    log_likelihoods = {}
    weights = weights or {}
    methods = methods or {}

    actual_keys = set(actual.keys())
    if not actual_keys <= set(weights.keys()):
        missing = actual_keys - set(weights.keys())
        print(f"[WARN] Missing weights for: {missing}. Defaulting to 1.0.")

    for key in actual:
        try:
            v_obs = np.array(actual[key], dtype=float)
            v_sim = np.array(predicted[key], dtype=float)
            v_sim = np.clip(v_sim, 1e-6, None)

            if v_obs.shape != v_sim.shape:
                sc.printyellow(f"[WARN] Shape mismatch on '{key}': {v_obs.shape} vs {v_sim.shape}")
                continue

            method = methods.get(key)
            if not method:
                method = "dirichlet_multinomial" if len(v_obs) > 1 else "poisson"

            if method == "poisson":
                logp = poisson.logpmf(v_obs, v_sim)
                ll = logp.sum()

            elif method == "neg_binomial":
                r = dispersion
                p = r / (r + v_sim)
                logp = nbinom.logpmf(v_obs, r, p)
                ll = logp.sum()

            elif method == "dirichlet_multinomial":
                # alpha = v_obs + 1
                # x = np.round(v_sim).astype(float)
                a_x = v_obs
                e_x = v_sim
                n = e_x.sum()

                ll = dirichlet_multinomial.logpmf(x=e_x, n=n, alpha=a_x + 1)

            else:
                raise ValueError(f"Unknown method '{method}'")

            weight = weights.get(key, 1.0)
            n = len(v_obs)
            normalizer = n if norm_by_n and method != "dirichlet_multinomial" else 1
            neg_ll = -1.0 * weight * ll / normalizer
            log_likelihoods[key] = neg_ll

        except Exception as e:
            print(f"[ERROR] Skipping '{key}' due to: {e}")

    log_likelihoods["total_log_likelihood"] = sum(log_likelihoods.values())
    return log_likelihoods
