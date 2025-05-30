import sys
from pathlib import Path

import cloud_calib_config as cfg
import optuna

sys.path.append(str(Path(__file__).resolve().parent.parent))
from report import plot_likelihoods
from report import plot_runtimes
from report import plot_stuff
from report import plot_targets
from report import save_study_results


def main():
    try:
        print(f"📊 Loading study '{cfg.study_name}'...")
        storage_url = "mysql+pymysql://optuna:superSecretPassword@mysql.default.svc.cluster.local:3306/optuna"
        study = optuna.load_study(study_name=cfg.study_name, storage=storage_url)
        study.storage_url = cfg.storage_url
        study.study_name = cfg.study_name

        results_path = Path("results") / cfg.study_name
        results_path.mkdir(parents=True, exist_ok=True)

        print("💾 Saving results...")
        save_study_results(study, output_dir=results_path)

        print("📈 Plotting results...")
        plot_stuff(cfg.study_name, study.storage_url, output_dir=results_path)

        print("📊 Plotting target comparisons...")
        plot_targets(study, output_dir=results_path)

        print("Plotting runtimes...")
        plot_runtimes(study, output_dir=results_path)

        print("Plotting likelihoods...")
        plot_likelihoods(study, output_dir=Path(results_path), use_log=True)

    finally:
        print("🎉Done!")


if __name__ == "__main__":
    main()
