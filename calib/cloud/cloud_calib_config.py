# calib_job_config.py
from pathlib import Path

import yaml

study_name = "calib_nigeria_20250423_fullpop"
num_trials = 1
parallelism = 4
completions = 20  # The total number of pods (i.e., jobs) that need to successfully complete before the job is considered "done"
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
