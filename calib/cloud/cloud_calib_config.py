# calib_job_config.py
from pathlib import Path

import yaml

# ------------------- USER CONFIGS -------------------

study_name = "calib_nigeria_3y_underwt_gravity_2018_20250522"
model_config = "config_nigeria_3y_underwt_gravity_2018.yaml"
calib_config = "r0_k_ssn_gravity.yaml"
fit_function = "log_likelihood"
num_trials = 1  # Number of trials to run per pod
n_replicates = 1  # Number of replicates to run for each trial
parallelism = 50  # The number of pods (i.e., jobs) to run in parallel
completions = 1000  # The total number of pods (i.e., jobs) that need to successfully complete before the job is considered "done"

# ---------------------------------------------------

# Default settings
namespace = "default"
job_name = "laser-polio-worker-sk"
image = "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest"
storage_url="mysql+pymysql://optuna:superSecretPassword@localhost:3306/optunaDatabase"

