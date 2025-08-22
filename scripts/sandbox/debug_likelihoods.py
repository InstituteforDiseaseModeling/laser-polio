import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from scipy.stats import poisson

# Solution seems to be to use DM as usual, but then add Mag(Total) as well, likely with lots of weight on Mag(Total).

# ------------------------------------------------------------
#  Set up and plot the data
# ------------------------------------------------------------

# Your data
actual = [0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 0]
preds = {
    "match": [0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 0],
    "sametime_low": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    "sametime_high": [0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 0],
    "difftime": [0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 0, 0],
    "longertime": [0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 0],
    "longertime_high": [0, 0, 0, 0, 0, 10, 10, 10, 100, 100, 100, 0],
}


def plot_actual_vs_preds(
    actual,
    preds: dict[str, list | np.ndarray],
    *,
    title: str = "Actual vs Predicted",
    outfile: str | None = None,
    common_ymax: float | None = None,
    ymax_pad_frac: float = 0.05,
    xlabel: str = "time (index)",
    ylabel: str = "cases",
):
    """
    Plot actual vs predicted time series as small multiples (one subplot per prediction).
    Each subplot title is "Actual vs. <pred name>" and the predicted line is orange.
    """
    names = list(preds.keys())

    actual = np.asarray(actual, dtype=float)
    x = np.arange(len(actual))
    for k in names:
        yk = np.asarray(preds[k], dtype=float)
        if yk.shape != actual.shape:
            raise ValueError(f"Prediction '{k}' has shape {yk.shape}, expected {actual.shape}.")

    if common_ymax is None:
        ymax_data = max(np.nanmax(actual), max(np.nanmax(np.asarray(preds[k], dtype=float)) for k in names))
        ymax_data = 0.0 if np.isnan(ymax_data) else float(ymax_data)
        common_ymax = (1.0 + max(0.0, float(ymax_pad_frac))) * ymax_data

    rows = len(names)
    fig, axes = plt.subplots(rows, 1, figsize=(10, max(2.2 * rows, 2.2)), sharex=True, constrained_layout=True)
    if rows == 1:
        axes = [axes]

    axes_ts = []
    for r, name in enumerate(names):
        ax = axes[r]
        ax.plot(x, actual, label="actual", color="black", lw=2)
        ax.plot(x, preds[name], label="predicted", lw=2, alpha=0.9, color="tab:orange")  # ← orange
        ax.set_ylim(0, common_ymax)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_title(f"Actual vs. {name}")  # ← per-plot title
        if r == 0:
            ax.legend(loc="upper left", ncols=2, fontsize=9, frameon=True)
        if r == rows - 1:
            ax.set_xlabel(xlabel)
        axes_ts.append(ax)

    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
    return fig, axes_ts


fig, axes = plot_actual_vs_preds(
    actual, preds, title="Actual vs Predicted (small multiples)", outfile="results/dirichlet_multinomial_actual_vs_preds.png"
)


# ------------------------------------------------------------
#  Scoring functions
# ------------------------------------------------------------


# OBJECTIVE 1: Shape (using DM)
def dm_logp(x_counts, pred_counts, eps=1e-12):
    """Assess shape similarity using Dirichlet-Multinomial"""
    x = np.asarray(x_counts, dtype=int)
    n = int(x.sum())
    alpha = np.clip(np.asarray(pred_counts, dtype=float), eps, None) + 1.0
    return float(sps.dirichlet_multinomial.logpmf(x=x, n=n, alpha=alpha))


def dm_logp_fixed_tau(x_counts, pred_counts, rho=1.0, eps=1e-12):
    """
    Calculate log likelihood using Dirichlet-multinomial distribution with tau based on observed total.
    """
    x = np.asarray(x_counts, dtype=int)
    n = int(x.sum())
    if n == 0:
        return 0.0  # P(X=0|DM) = 1 ⇒ logp = 0
    p = np.clip(np.asarray(pred_counts, dtype=float), eps, None)
    p /= np.sum(p)
    tau = rho * n
    alpha = tau * p  # Tau tied to observed total which is constant across sims
    return float(sps.dirichlet_multinomial.logpmf(x=x, n=n, alpha=alpha))


# OBJECTIVE 2: Total magnitude
def magnitude_score_total(x_counts, pred_counts):
    """
    Simple magnitude assessment based on total counts.
    Returns log probability of observed total given predicted total.
    """
    actual_total = np.sum(x_counts)
    pred_total = np.sum(pred_counts)

    # Use Poisson or Normal likelihood for the total
    if pred_total > 0:
        # Poisson likelihood for total count
        return poisson.logpmf(actual_total, pred_total)
    else:
        return -np.inf if actual_total > 0 else 0


# OBJECTIVE 3: Position-wise magnitude
def magnitude_score_positions(x_counts, pred_counts, eps=1e-12):
    """
    Assess magnitude at each position using scale-sensitive metric.
    Uses squared log ratio to penalize magnitude differences.
    """
    x = np.asarray(x_counts, dtype=float)
    pred = np.asarray(pred_counts, dtype=float)

    # Add small epsilon to avoid log(0)
    x_safe = x + eps
    pred_safe = pred + eps

    # Calculate squared log ratios (symmetric, scale-sensitive)
    log_ratios = np.log(x_safe / pred_safe)

    # Return negative mean squared log ratio
    return -np.mean(log_ratios**2)


logps_shape = {name: dm_logp(actual, yhat) for name, yhat in preds.items()}
logps_shape_fixed_tau = {name: dm_logp_fixed_tau(actual, yhat) for name, yhat in preds.items()}
logps_mag_total = {name: magnitude_score_total(actual, yhat) for name, yhat in preds.items()}
logps_shape_mag_total = {name: logps_shape[name] + logps_mag_total[name] for name in preds.keys()}
logps_shape_fixed_tau_mag_total = {name: logps_shape_fixed_tau[name] + logps_mag_total[name] for name in preds.keys()}


# ------------------------------------------------------------
# Make a table of the logps
# ------------------------------------------------------------


# EVALUATION
print("SEPARATE OBJECTIVES ANALYSIS")
print("=" * 60)


# Calculate all scores
results = {}
for name, pred in preds.items():
    results[name] = {
        "shape_dm": dm_logp(actual, pred),
        "shape_dm_fixed_tau": dm_logp_fixed_tau(actual, pred),
        "mag_total": magnitude_score_total(actual, pred),
        "dm_mag_total": dm_logp(actual, pred) + magnitude_score_total(actual, pred),
        "dm_mag_total_fixed_tau": dm_logp_fixed_tau(actual, pred) + magnitude_score_total(actual, pred),
    }

# Display results in a nice table
print(f"{'Prediction':<20} {'DM(current)':<12} {'DM(fixed tau)':<12} {'Mag(Total)':<12} {'DM(cur) + Mag':<12} {'DM(fixed) + Mag':<12}")
print("-" * 80)
for name in preds.keys():
    r = results[name]
    print(
        f"{name:<20} {r['shape_dm']:<12.4f} {r['shape_dm_fixed_tau']:<12.4f} {r['mag_total']:<12.4f} {r['dm_mag_total']:<12.4f} {r['dm_mag_total_fixed_tau']:<12.4f}"
    )


# # Calculate all scores
# results = {}
# for name, pred in preds.items():
#     results[name] = {
#         "shape_dm": dm_logp(actual, pred),
#         "mag_total": magnitude_score_total(actual, pred),
#         "mag_positions": magnitude_score_positions(actual, pred),
#         "dm_mag_total": dm_logp(actual, pred) + magnitude_score_total(actual, pred),
#         "dm_mag_positions": dm_logp(actual, pred) + magnitude_score_positions(actual, pred),
#         "dm_mag_total_mag_positions": dm_logp(actual, pred) + magnitude_score_total(actual, pred) + magnitude_score_positions(actual, pred),
#     }

# # Display results in a nice table
# print(f"{'Prediction':<20} {'Shape(DM)':<12} {'Mag(Total)':<12} {'Mag(Pos)':<12} {'DM + Mag(Total)':<12} {'DM + Mag(Pos)':<12} {'DM + Mag(Total) + Mag(Pos)':<12}")
# print("-" * 80)
# for name in preds.keys():
#     r = results[name]
#     print(
#         f"{name:<20} {r['shape_dm']:<12.4f} {r['mag_total']:<12.4f} "
#         f"{r['mag_positions']:<12.4f} {r['dm_mag_total']:<12.4f} {r['dm_mag_positions']:<12.4f} {r['dm_mag_total_mag_positions']:<12.4f}"
#     )
