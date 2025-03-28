import os

import click
import optuna
from objective import objective  # Ensure this is correctly defined elsewhere


@click.command()
@click.option("--study-name", default="laser_polio_test", help="Name of the Optuna study to load or create.")
@click.option("--num-trials", default=1, type=int, help="Number of trials for the optimization.")
def run_worker(study_name, num_trials):
    """Run an Optuna worker that performs optimization trials."""

    if os.getenv("STORAGE_URL"):
        storage_url = os.getenv("STORAGE_URL")
    else:
        # Construct the storage URL from environment variables
        # storage_url = "mysql+pymysql://{}:{}@optuna-mysql:3306/{}".format(
        storage_url = "mysql://{}:{}@mysql:3306/{}".format(
            os.getenv("MYSQL_USER", "root"), os.getenv("MYSQL_PASSWORD", ""), os.getenv("MYSQL_DB", "optuna_db")
        )

    print(f"storage_url={storage_url}")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception:
        print(f"Study '{study_name}' not found. Creating a new study.")
        study = optuna.create_study(study_name=study_name, storage=storage_url)

    # Run the trials
    study.optimize(objective, n_trials=num_trials)


if __name__ == "__main__":
    run_worker()
