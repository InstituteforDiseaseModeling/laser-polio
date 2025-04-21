import json
from datetime import datetime
from pathlib import Path

import optuna
import optuna.visualization as vis


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
    print("\nBest Trial:")
    print(f"  Value: {best.value}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save trials dataframe
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(output_dir / csv_name, index=False)

    # Save best params
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=4)

    # Save metadata
    metadata = dict(study.user_attrs)  # copy user_attrs
    metadata["timestamp"] = metadata.get("timestamp") or datetime.now().isoformat()  # noqa: DTZ005
    metadata["study_name"] = study.study_name
    metadata["storage_url"] = study.storage_url
    with open(output_dir / "study_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Study results saved to '{output_dir}'")


def plot_stuff(study_name, storage_url):
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    vis.plot_optimization_history(study).show()
    try:
        vis.plot_param_importances(study).show()
    except Exception as ex:
        print("Exception trying to plot param importances; this usually happens when all trials have produced the same score.")
        print(str(ex))
    vis.plot_slice(study).show()
    vis.plot_contour(study, params=["r0", "gravity_k"]).show()  # pick any 2 params
