# calib_job_config.py
from pathlib import Path

import yaml

# ------------------- USER CONFIGS -------------------

study_name = "calib_nigeria_6y_pim_r0_scalars_20250519"
model_config = "config_nigeria_6y_pim_r0_scalars.yaml"
calib_config = "r0_k_ssn.yaml"
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

# Define the path to the YAML file with the storage URL from the docs
storage_path = Path("calib/cloud/local_storage.yaml")

# Try loading the storage URL from YAML, fallback to env var
storage_url = None
if storage_path.exists():
    storage = yaml.safe_load(storage_path.read_text())
    storage_url = storage.get("storage_url")
# Safety check
print(f"Storage URL: {storage_url}")
if storage_url is None:
    raise RuntimeError("Missing STORAGE_URL in local_storage.yaml")
