import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from matplotlib.gridspec import GridSpec

# --- your data ---
actual = [0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 0]
preds = {
    "sametime_low": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    "sametime_high": [0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 0],
    "match": [0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 0],
    "difftime": [0, 0, 0, 0, 0, 10, 10, 10, 0, 0, 0, 0],
    "longertime": [0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 0],
    "longertime_high": [0, 0, 0, 0, 0, 10, 10, 10, 100, 100, 100, 0],
}


# --- log-likelihood (Dirichlet-multinomial) ---
def dm_logp(x_counts, pred_counts, eps=1e-12):
    x = np.asarray(x_counts, dtype=int)
    n = int(x.sum())
    alpha = np.clip(np.asarray(pred_counts, dtype=float), eps, None) + 1.0
    return float(sps.dirichlet_multinomial.logpmf(x=x, n=n, alpha=alpha))


logps = {name: dm_logp(actual, yhat) for name, yhat in preds.items()}

# --- plotting ---
names = list(preds.keys())
# Sort by logp (best at top of bar chart)
order = np.argsort([logps[n] for n in names])[::-1]
names_sorted = [names[i] for i in order]
logps_sorted = [logps[n] for n in names_sorted]

# Shared y-limit across all time-series panels
ymax = 1.05 * max(np.max(actual), max(np.max(preds[n]) for n in names))

rows = len(names)
fig = plt.figure(figsize=(12, 2.2 * rows + 2))
gs = GridSpec(rows, 2, width_ratios=[3.0, 1.4], hspace=0.5, wspace=0.3)

# Left column: one row per prediction
x = np.arange(len(actual))
for r, name in enumerate(names_sorted):
    ax = fig.add_subplot(gs[r, 0])
    ax.plot(x, actual, label="actual", color="black", lw=2)
    ax.plot(x, preds[name], label=name, lw=2, alpha=0.8)
    ax.set_ylim(0, ymax)
    if r == rows - 1:
        ax.set_xlabel("time (index)")
    ax.set_ylabel("cases")
    ax.grid(alpha=0.3)
    # annotate logp on the panel
    ax.text(
        0.98,
        0.95,
        f"logp = {logps[name]:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    if r == 0:
        ax.legend(loc="upper left", ncols=2, fontsize=9, frameon=True)

# Right column: bar chart of logps
axb = fig.add_subplot(gs[:, 1])
axb.barh(names_sorted, logps_sorted)
axb.invert_yaxis()  # best at top
axb.set_xlabel("Dirichlet-multinomial log-likelihood")
axb.set_title("Model fit (higher is better)")
# highlight best
best_val = logps_sorted[0]
axb.axvline(best_val, ls="--", lw=1)
for y, v in enumerate(logps_sorted):
    axb.text(v, y, f" {v:.2f}", va="center", ha="left", fontsize=9)

fig.suptitle("Actual vs Predicted (with log-likelihood)", y=0.995, fontsize=13)
fig.savefig("results/debug_dirichlet.png")
fig.show()
