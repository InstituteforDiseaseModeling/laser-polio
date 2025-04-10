import subprocess

import yaml
from kubernetes import client
from kubernetes import config

# Load Kubernetes config
config.load_kube_config()

# Kubernetes API client
batch_v1 = client.BatchV1Api()

# Load and process the manifest YAML file
yaml_file = "laser-study-creator-deploy-manifests.yaml"

with open(yaml_file) as f:
    manifest = yaml.safe_load(f)


def get_laser_polio_deps():
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "cat",
                "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest",
                "/app/laser_polio_deps.txt",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        deps_output = result.stdout
        with open("laser_polio_deps.txt", "w") as f:
            f.write(deps_output)
        print("✅ laser_polio_deps.txt retrieved and saved locally.")

    except subprocess.CalledProcessError as e:
        print("❌ Failed to retrieve laser_polio_deps.txt:")
        print(e.stderr)


# Call the function before submitting the job
get_laser_polio_deps()

# Replace string variables in the manifest
env_vars = {
    "STUDY_NAME": "my_study",
    "STORAGE_URL": "mysql+pymysql://optuna:superSecretPassword@optuna-mysql:3306/optunaDatabase",
}


# Function to recursively replace env variables in the YAML structure
def replace_env_vars(obj):
    if isinstance(obj, dict):
        return {k: replace_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_env_vars(i) for i in obj]
    elif isinstance(obj, str):
        for key, val in env_vars.items():
            obj = obj.replace(f"${{{key}}}", val)
        return obj
    return obj


manifest = replace_env_vars(manifest)

# Apply the job definition
namespace = "default"  # Change if using a different namespace

try:
    response = batch_v1.create_namespaced_job(namespace=namespace, body=manifest)
    print(f"Job {response.metadata.name} created successfully.")
except client.exceptions.ApiException as e:
    print(f"Error applying the job: {e}")
