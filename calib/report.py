import json
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
import sciris as sc
import yaml


def save_study_results(study, output_dir: Path, csv_name: str = "trials.csv"):
    """
    Saves the essential outputs of an Optuna study (best params, metadata,
    trials data) into the given directory. Caller is responsible for loading
    the study and doing anything else (like copying model configs).

    :param study: An already-loaded Optuna study object.
    :param output_dir: Where to write all output files.
    :param csv_name: Optional CSV filename for trial data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print a brief best-trial summary
    best = study.best_trial
    sc.printcyan("\nBest Trial:")
    print(f"  Value: {best.value}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save trials dataframe
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    df.to_csv(output_dir / csv_name, index=False)

    # Save best params
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=4)

    # Save metadata
    metadata = dict(study.user_attrs)  # copy user_attrs
    metadata["timestamp"] = metadata.get("timestamp") or datetime.now().isoformat()  # noqa: DTZ005
    metadata["study_name"] = study.study_name
    metadata["storage_url"] = study.storage_url
    try:
        metadata["laser_polio_git_info"] = sc.gitinfo()
    except Exception:
        metadata["laser_polio_git_info"] = "Unavailable (no .git info in Docker)"
    with open(output_dir / "study_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Study results saved to '{output_dir}'")


def plot_stuff(study_name, storage_url, output_dir=None):
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # Default output directory to current working dir if not provided
    output_dir = Path(output_dir) if output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving Optuna plots to: {output_dir.resolve()}")

    # Optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.write_html(output_dir / "plot_opt_history.html")

    # Param importances
    try:
        fig2 = vis.plot_param_importances(study)
        fig2.write_html(output_dir / "plot_param_importances.html")
    except Exception as ex:
        print("[WARN] Could not plot param importances:", ex)

    # Slice plot
    fig3 = vis.plot_slice(study)
    fig3.write_html(output_dir / "plot_slice.html")

    # Contour plot — feel free to customize parameters
    fig4 = vis.plot_contour(study, params=["r0", "radiation_k"])
    fig4.write_html(output_dir / "plot_contour.html")


def plot_targets(study, output_dir=None, shp=None):
    best = study.best_trial
    actual = best.user_attrs["actual"]
    preds = best.user_attrs["predicted"]

    # Load region group labels from study_metadata.json
    metadata_path = Path(output_dir) / "study_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"study_metadata.json not found at {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    model_config = metadata.get("model_config", {})
    region_groups = model_config.get("summary_config", {}).get("region_groups", {})
    region_labels = list(region_groups.keys())

    # Define consistent colors
    n_reps = len(preds)
    labels = ["Actual"] + [f"Rep {i + 1}" for i in range(n_reps)]
    cmap = cm.get_cmap("Dark2")  # tab10
    color_map = {label: cmap(i) for i, label in enumerate(labels)}

    # Total Infected
    plt.figure()
    plt.title("Total Infected")
    width = 0.2
    x = np.arange(len(labels))
    values = [actual["total_infected"][0]] + [rep["total_infected"][0] for rep in preds]
    plt.bar(x, values, width=width, color=[color_map[label] for label in labels])
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Cases")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_total_infected_comparison.png")

    # Yearly Cases
    years = list(range(2018, 2018 + len(actual["yearly_cases"])))
    plt.figure()
    plt.title("Yearly Cases")
    plt.plot(years, actual["yearly_cases"], "o-", label="Actual", color=color_map["Actual"], linewidth=2)
    for i, rep in enumerate(preds):
        label = f"Rep {i + 1}"
        plt.plot(years, rep["yearly_cases"], "o-", label=label, color=color_map[label])
    plt.xlabel("Year")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_yearly_cases_comparison.png")
    # plt.show()

    # Monthly Cases
    months = list(range(1, 1 + len(actual["monthly_cases"])))
    plt.figure()
    plt.title("Monthly Cases")
    plt.plot(months, actual["monthly_cases"], "o-", label="Actual", color=color_map["Actual"], linewidth=2)
    for i, rep in enumerate(preds):
        label = f"Rep {i + 1}"
        plt.plot(months, rep["monthly_cases"], "o-", label=label, color=color_map[label])
    plt.xlabel("Month")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_monthly_cases_comparison.png")
    # plt.show()

    # Monthly Timeseries Cases
    n_months = len(actual["monthly_timeseries"])
    months_series = pd.date_range(start="2018-01-01", periods=n_months, freq="MS")
    plt.figure()
    plt.title("Monthly Timeseries")
    plt.plot(months_series, actual["monthly_timeseries"], "o-", label="Actual", color=color_map["Actual"], linewidth=2)
    for i, rep in enumerate(preds):
        label = f"Rep {i + 1}"
        plt.plot(months_series, rep["monthly_timeseries"], "o-", label=label, color=color_map[label])
    plt.xlabel("Month")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(output_dir / "plot_best_monthly_timeseries_comparison.png")
    plt.show()

    # Regional Cases (bar plot)
    x = np.arange(len(region_labels))
    width = 0.1
    plt.figure()
    plt.title("Regional Cases")
    plt.bar(x, actual["regional_cases"], width, label="Actual", color=color_map["Actual"])
    for i, rep in enumerate(preds):
        label = f"Rep {i + 1}"
        plt.bar(x + (i + 1) * width, rep["regional_cases"], width, label=f"Rep {i + 1}", color=color_map[label])
    plt.xticks(x + width * (len(preds) // 2), region_labels)
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_regional_cases_comparison.png")
    # plt.show()

    # Total Nodes with Cases
    plt.figure()
    plt.title("Total Nodes with Cases")
    width = 0.2
    x = np.arange(1 + len(preds))
    values = [actual["total_nodes_with_cases"][0]] + [rep["total_nodes_with_cases"][0] for rep in preds]
    labels = ["Actual"] + [f"Rep {i + 1}" for i in range(len(preds))]
    plt.bar(x, values, width=width, color=[color_map[lbl] for lbl in labels])
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Nodes")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_total_nodes_with_cases.png")

    # Monthly Nodes with Cases
    n_months = len(actual["monthly_nodes_with_cases"])
    months = list(range(1, n_months + 1))
    plt.figure()
    plt.title("Monthly Nodes with Cases")
    plt.plot(months, actual["monthly_nodes_with_cases"], "o-", label="Actual", color=color_map["Actual"], linewidth=2)
    for i, rep in enumerate(preds):
        label = f"Rep {i + 1}"
        plt.plot(months, rep["monthly_nodes_with_cases"], "o-", label=f"Rep {i + 1}", color=color_map[label])
    plt.xlabel("Month")
    plt.ylabel("Number of Nodes with ≥1 Case")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_monthly_nodes_with_cases.png")

    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import Normalize
    # from matplotlib.cm import ScalarMappable
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # # Plot monthly timeseries with annual maps
    # if shp is not None:
    #     months_series
    #     years = sorted(set(months_series.year))

    #     # Compute yearly sum per node
    #     months_df = pd.DataFrame(monthly_cases_per_node, index=months_series)
    #     annual_cases_by_node = {year: months_df.loc[str(year)].sum(axis=0).values for year in years}

    #     # Normalize color scale across years
    #     global_min = min(np.min(v) for v in annual_cases_by_node.values())
    #     global_max = max(np.max(v) for v in annual_cases_by_node.values())
    #     norm = Normalize(vmin=global_min, vmax=global_max)
    #     cmap = truncate_colormap(cmap_name, minval=0.1, maxval=0.9)

    #     # Create figure
    #     fig, ax = plt.subplots(figsize=(16, 6))
    #     ax.plot(months_series, actual_timeseries, "o-", label="Actual", linewidth=2)
    #     ax.set_title("Monthly Timeseries with Annual Maps")
    #     ax.set_xlabel("Month")
    #     ax.set_ylabel("Cases")
    #     ax.legend()

    #     # Plot inset maps aligned under each year
    #     for i, year in enumerate(years):
    #         center_date = pd.Timestamp(f"{year}-07-01")  # middle of the year
    #         x_pos = center_date

    #         # Convert data coords to figure fraction
    #         trans = ax.transData.transform((x_pos.toordinal(), ax.get_ylim()[0]))
    #         inv = fig.transFigure.inverted().transform(trans)
    #         x_fig, y_fig = inv

    #         # Add inset axes
    #         axins = inset_axes(
    #             ax,
    #             width="5%", height="20%",  # relative to figure
    #             bbox_to_anchor=(x_fig - 0.025, 0.02, 0.05, 0.15),
    #             bbox_transform=fig.transFigure,
    #             loc="lower left",
    #             borderpad=0,
    #         )

    #         shp["cases"] = annual_cases_by_node[year]
    #         shp.plot(
    #             column="cases",
    #             ax=axins,
    #             cmap=cmap,
    #             norm=norm,
    #             edgecolor="white",
    #             linewidth=0.1,
    #             legend=False
    #         )
    #         axins.set_title(f"{year}", fontsize=8)
    #         axins.set_axis_off()

    #     # Shared colorbar
    #     sm = ScalarMappable(cmap=cmap, norm=norm)
    #     sm._A = []
    #     cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.01)
    #     cbar.set_label("Annual Node-Level Cases")

    #     plt.tight_layout()
    #     plt.show()


def plot_runtimes(study, output_dir=None):
    # Collect runtimes of completed trials
    durations = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.datetime_start and trial.datetime_complete:
            duration = trial.datetime_complete - trial.datetime_start
            durations.append(duration.total_seconds() / 60)  # Convert to minutes

    # Plot histogram with mean runtime
    if durations:
        avg_runtime = sum(durations) / len(durations)

        plt.figure(figsize=(8, 5))
        plt.hist(durations, bins=20, edgecolor="black", alpha=0.75)

        # Add vertical dashed mean line
        plt.axvline(avg_runtime, color="red", linestyle="--", linewidth=2, label=f"Mean = {avg_runtime:.2f} min")

        # Annotated title
        plt.title("Histogram of Optuna Trial Runtimes")
        plt.xlabel("Trial Runtime (minutes)")
        plt.ylabel("Number of Trials")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_runtimes.png")
        # plt.show()
    else:
        print("No completed trials with valid timestamps to plot.")


def load_region_group_labels(model_config_path):
    with open(model_config_path) as f:
        config = yaml.safe_load(f)
    region_groups = config.get("summary_config", {}).get("region_groups", {})
    return list(region_groups.keys())
