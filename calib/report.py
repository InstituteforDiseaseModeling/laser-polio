import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization as vis
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


def plot_targets(study, output_dir=None):
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

    # Total Infected (scatter plot)
    plt.figure()
    plt.title("Total Infected")
    plt.xticks([0], ["total"])
    plt.plot(0, actual["total_infected"][0], "o", label="Actual")
    for i, rep in enumerate(preds):
        plt.plot(0, rep["total_infected"][0], "x", label=f"Predicted rep {i + 1}")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_total_infected_comparison.png")
    # plt.show()

    # Yearly Cases
    years = list(range(2018, 2018 + len(actual["yearly_cases"])))
    plt.figure()
    plt.title("Yearly Cases")
    plt.plot(years, actual["yearly_cases"], "o-", label="Actual", linewidth=2)
    for i, rep in enumerate(preds):
        plt.plot(years, rep["yearly_cases"], "o-", label=f"Rep {i + 1}")
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
    plt.plot(months, actual["monthly_cases"], "o-", label="Actual", linewidth=2)
    for i, rep in enumerate(preds):
        plt.plot(months, rep["monthly_cases"], "o-", label=f"Rep {i + 1}")
    plt.xlabel("Month")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_monthly_cases_comparison.png")
    # plt.show()

    # Regional Cases (bar plot)
    x = np.arange(len(region_labels))
    width = 0.1
    plt.figure()
    plt.title("Regional Cases")
    plt.bar(x, actual["regional_cases"], width, label="Actual")
    for i, rep in enumerate(preds):
        plt.bar(x + (i + 1) * width, rep["regional_cases"], width, label=f"Rep {i + 1}")
    plt.xticks(x + width * (len(preds) // 2), region_labels)
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_best_regional_cases_comparison.png")
    # plt.show()


def load_region_group_labels(model_config_path):
    with open(model_config_path) as f:
        config = yaml.safe_load(f)
    region_groups = config.get("summary_config", {}).get("region_groups", {})
    return list(region_groups.keys())
