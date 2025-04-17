# calib_job_config.py

study_name = "test_polio_calib_fixed"
num_trials = 1
parallelism = 4
completions = 20
namespace = "default"
job_name = "laser-polio-worker-jb"
image = "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest"

# Placeholder
storage_url = "REPLACE_WITH_VALUE_FROM_DOCS"

if storage_url == "REPLACE_WITH_VALUE_FROM_DOCS":
    raise RuntimeError(
        "\n[ERROR] You must set `storage_url` in calib_job_config.py!\n"
        "Follow the instructions in the documentation to get the correct value.\n"
        "Hint: It's probably an Azure blob storage URL with credentials.\n"
    )
