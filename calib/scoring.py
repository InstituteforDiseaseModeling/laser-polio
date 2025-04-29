import numpy as np
import sciris as sc
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


def compute_log_likelihood_fit(actual, predicted, method="poisson", dispersion=1.0, weights=None):
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
    log_likelihood = 0.0
    weights = weights or {}

    for key in actual:
        if key not in predicted:
            print(f"[WARN] Key missing in predicted: {key}")
            continue

        try:
            v_obs = np.array(actual[key], dtype=float)
            v_sim = np.array(predicted[key], dtype=float)
            v_sim = np.clip(v_sim, 1e-6, None)  # Prevent log(0) in Poisson

            if v_obs.shape != v_sim.shape:
                sc.printyellow(f"[WARN] Shape mismatch on '{key}': {v_obs.shape} vs {v_sim.shape}")
                continue

            if method == "poisson":
                logp = poisson.logpmf(v_obs, v_sim)
            elif method == "neg_binomial":
                # NB parameterization via mean (mu) and dispersion (r)
                # r = dispersion; p = r / (r + mu)
                mu = v_sim
                r = dispersion
                p = r / (r + mu)
                logp = nbinom.logpmf(v_obs, r, p)
            else:
                raise ValueError(f"Unknown method '{method}'")

            # Sum log-likelihoods, but normalize by number of observations (e.g., total_infected has 1 value, while monthly_cases has 12)
            n = len(logp)
            weight = weights.get(key, 1)
            log_likelihood += weight * logp.sum() / n

        except Exception as e:
            print(f"[ERROR] Skipping '{key}' due to: {e}")

    return -log_likelihood  # NEGATE for Optuna
