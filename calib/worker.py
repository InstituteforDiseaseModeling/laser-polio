import os
import yaml

import click
import optuna
import objective
#from objective import objective  # Ensure this is correctly defined elsewhere


@click.command()
@click.option("--study-name", default="laser_polio_test", help="Name of the Optuna study to load or create.")
@click.option("--num-trials", default=1, type=int, help="Number of trials for the optimization.")
@click.option("--calib-pars", default="calib_pars.yaml", type=str, help="Config file with params to calibrate.")
@click.option("--config-pars", default="config.yaml", type=str, help="Config file with base params for model.")
def run_worker(study_name, num_trials, calib_pars, config_pars):
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

    # Load calibration parameters from YAML
    with open(calib_pars, "r") as f:
        calib_pars_dict = yaml.safe_load(f)

    objective.calib_pars = calib_pars_dict

    # Load calibration parameters from YAML
    with open(config_pars, "r") as f:
        config_pars_dict = yaml.safe_load(f)

    objective.config_pars = config_pars_dict

    # Set study-level metadata for reproducibility
    metadata = calib_pars_dict.get("metadata", {})
    for key, value in metadata.items():
        study.set_user_attr(key, value)

    # Store the parameter specification itself
    study.set_user_attr("parameter_spec", calib_pars_dict.get("parameters", {}))

    # Run the trials
    study.optimize(objective.objective, n_trials=num_trials)

if __name__ == "__main__":
    run_worker()
