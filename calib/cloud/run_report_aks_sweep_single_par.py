"""
Overlay `cases_by_month` across trials while holding all params near-best
except one "sweep" parameter (e.g., seasonal_amplitude).

- Selects near-best trials per bin of the sweep param
- Overlays `cases_by_month` for N trials from each bin on separate figures
- NEW: All figures share the same ymax (with small headroom)
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cloud_calib_config as cfg
import matplotlib.pyplot as plt
import numpy as np
import optuna

# ------------------- USER CONFIGS -------------------
USER = {
    "study_name": None,  # If None, use cloud_calib_config.study_name
    "sweep_param": "seasonal_peak_doy",
    "n_bins": 3,  # Number of bins across the sweep param
    "n_per_bin": 10,  # Number of trials to plot per bin
    "output_root": None,  # If None -> results/<study_name>/sweep_<param>/
    "common_ymax_pad_frac": 0.05,  # Add 5% headroom above the global max
}
# ----------------- END USER CONFIGS -----------------


def port_forward():
    print("🔌 Setting up port forwarding to MySQL...")
    pf = subprocess.Popen(["kubectl", "port-forward", "mysql-0", "3307:3306"])
    time.sleep(3)  # wait for port-forward to take effect
    return pf


def main():
    pf_process = port_forward()
    try:
        study_name = USER["study_name"] or cfg.study_name
        print(f"📊 Loading study '{study_name}'...")
        study = optuna.load_study(study_name=study_name, storage=cfg.storage_url)

        sweep_par = USER["sweep_param"]
        results_path = Path(USER["output_root"] or (Path("results") / study_name / f"sweep_{sweep_par}"))
        results_path.mkdir(parents=True, exist_ok=True)

        # Fetch the trials data (COMPLETE only)
        trials = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs"))
        trials = trials[trials["state"] == "COMPLETE"].copy()

        # Split the sweep param into equal-width value bins
        n_bins = int(USER["n_bins"])
        n_per_bin = int(USER["n_per_bin"])
        trials = trials.sort_values(by=f"params_{sweep_par}")
        values = trials[f"params_{sweep_par}"].to_numpy()
        lo, hi = float(values.min()), float(values.max())
        edges = np.linspace(lo, hi, n_bins + 1)

        # First pass: pick samples per bin and compute a global ymax across
        # ACTUAL and all SELECTED predicted curves for consistent axes.
        selections = []  # list of (bin_index, Index-of-selected-trial-numbers)
        global_ymax = 0.0

        for i in range(n_bins):
            # For the last bin, include the right edge so max value isn't dropped
            left = edges[i]
            right = edges[i + 1]
            if i < n_bins - 1:
                mask = (trials[f"params_{sweep_par}"] >= left) & (trials[f"params_{sweep_par}"] < right)
            else:
                mask = (trials[f"params_{sweep_par}"] >= left) & (trials[f"params_{sweep_par}"] <= right)
            bin_trials = trials[mask]

            if bin_trials.empty:
                print(f"⚠️  Bin {i} is empty. Skipping.")
                continue

            # Sample evenly across the value domain in this bin
            bin_values = bin_trials[f"params_{sweep_par}"].to_numpy()
            sel_idx, _ = sample_even_value_space(bin_values, k=n_per_bin, return_indices=True)
            # Guard against bins with fewer than k values
            sel_idx = np.asarray(sel_idx, dtype=int)
            sel_idx = sel_idx[(sel_idx >= 0) & (sel_idx < len(bin_trials))]
            bin_selected = bin_trials.iloc[sel_idx]
            selected_trial_numbers = bin_selected.index  # trial numbers are the DF index

            if selected_trial_numbers.empty:
                print(f"⚠️  No selectable trials in bin {i}.")
                continue

            selections.append((i, selected_trial_numbers))

            # Update global_ymax from this bin's actual + predicted curves
            # (Assumes actual/predicted stored in user_attrs per trial)
            # Use the first selected trial to pull 'actual' (expected to be the same across trials)
            first_tn = int(selected_trial_numbers[0])
            actual = study.trials[first_tn].user_attrs.get("actual", None)
            if isinstance(actual, dict) and "cases_by_month" in actual:
                ay = np.asarray(actual["cases_by_month"], dtype=float)
                if ay.size:
                    global_ymax = max(global_ymax, float(np.nanmax(ay)))

            best = study.best_trial
            best_y = np.asarray(best.user_attrs["predicted"][0]["cases_by_month"], dtype=float)
            if best_y.size:
                global_ymax = max(global_ymax, float(np.nanmax(best_y)))

            for tn in selected_trial_numbers:
                ua = study.trials[int(tn)].user_attrs
                pred_list = ua.get("predicted", None)
                if pred_list is None:
                    continue
                # Your structure shows predicted is a list, index 0 is a dict
                pred = pred_list[0] if isinstance(pred_list, (list, tuple)) else pred_list
                if isinstance(pred, dict) and "cases_by_month" in pred:
                    py = np.asarray(pred["cases_by_month"], dtype=float)
                    if py.size:
                        global_ymax = max(global_ymax, float(np.nanmax(py)))

        if global_ymax == 0.0:
            print("⚠️  Could not determine global_ymax (no data?). Proceeding with autoscale.")
            common_ymax = None
        else:
            pad = float(USER.get("common_ymax_pad_frac", 0.0))
            common_ymax = global_ymax * (1.0 + max(0.0, pad))
            print(f"📐 Using common ymax = {common_ymax:.4g} (base {global_ymax:.4g} + {pad * 100:.1f}% pad)")

        # Second pass: make the plots, applying the common ymax
        for i, selected_trial_numbers in selections:
            plot_cases_overlay(
                study=study,
                trial_numbers=selected_trial_numbers,
                sweep_par=sweep_par,
                title=f"Sweep over {sweep_par} - Bin {i}",
                output_path=results_path / f"sweep_{sweep_par}_bin_{i}.png",
                ymax=common_ymax,  # <— same ymax for all figures
            )

    finally:
        print("🧹 Cleaning up port forwarding...")
        try:
            pf_process.terminate()
        except Exception as e:
            print(f"⚠️  Warning: Could not terminate port-forward process: {e}")


def sample_even_value_space(values, k, include_first=True, include_last=True, return_indices=False):
    """
    Pick k samples spread evenly across the numeric range of `values`.
    Works even if values are highly concentrated in one region.
    Returns (indices, picked_values) if return_indices=True.
    """
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return (np.array([], dtype=int), vals) if return_indices else vals

    # If there are fewer points than k, just take all of them
    k = int(min(int(k), len(vals)))
    lo, hi = float(vals.min()), float(vals.max())

    # Build k target points in VALUE space
    if include_first and include_last:  # include both ends
        targets = np.linspace(lo, hi, k)
    elif include_first and not include_last:  # include lo, exclude hi
        step = (hi - lo) / k if k > 0 else 0.0
        targets = lo + step * np.arange(k)
    elif not include_first and include_last:  # exclude lo, include hi
        step = (hi - lo) / k if k > 0 else 0.0
        targets = lo + step * (np.arange(1, k + 1))
    else:  # exclude both ends
        targets = np.linspace(lo, hi, k + 2)[1:-1]

    # Greedy nearest-neighbor pick with de-dup
    avail_idx = np.arange(len(vals))
    avail_vals = vals.copy()
    chosen = []
    for t in targets:
        j = int(np.argmin(np.abs(avail_vals - t)))
        chosen.append(avail_idx[j])
        avail_vals = np.delete(avail_vals, j)
        avail_idx = np.delete(avail_idx, j)
        if avail_vals.size == 0:
            break

    chosen = np.array(chosen, dtype=int)
    return (chosen, values[chosen]) if return_indices else values[chosen]


def plot_cases_overlay(study, trial_numbers, sweep_par, title, output_path, ymax=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pull 'actual' from the first trial (assumed same across trials)
    first_tn = int(trial_numbers[0])
    actual = study.trials[first_tn].user_attrs.get("actual", None)
    if isinstance(actual, dict) and "cases_by_month" in actual:
        ay = np.asarray(actual["cases_by_month"], dtype=float)
        x = np.arange(len(ay))
        ax.plot(x, ay, alpha=0.8, label="actual", color="black", linewidth=3)
    else:
        ay = None

    # Pull 'best' calib values
    best = study.best_trial.user_attrs["predicted"][0]
    best_param = study.best_trial.params.get(sweep_par, None)
    if isinstance(best, dict) and "cases_by_month" in best:
        by = np.asarray(best["cases_by_month"], dtype=float)
        x = np.arange(len(by))
        ax.plot(x, by, alpha=0.8, label=f"best ({sweep_par}={best_param:.4g})", color="black", linestyle=":", linewidth=2)
    else:
        by = None

    # Overlay predictions
    for tn in trial_numbers:
        ua = study.trials[int(tn)].user_attrs
        pred_list = ua.get("predicted", None)
        pred = pred_list[0] if isinstance(pred_list, (list, tuple)) else pred_list
        if not (isinstance(pred, dict) and "cases_by_month" in pred):
            continue
        y = np.asarray(pred["cases_by_month"], dtype=float)
        x = np.arange(len(y))
        par_val = study.trials[int(tn)].params.get(sweep_par, None)
        lbl = f"trial {tn}" + (f", {sweep_par}={par_val:.4g}" if par_val is not None else "")
        ax.plot(x, y, alpha=0.6, label=lbl, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Months since start")
    ax.set_ylabel("Cases")
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.legend(loc="best", fontsize=8)

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"💾 Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
