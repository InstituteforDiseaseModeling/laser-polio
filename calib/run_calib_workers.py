import yaml
from kubernetes import client
from kubernetes import config

# Load local kubeconfig (assumes users have kubectl set up)
config.load_kube_config()

# Kubernetes API client
batch_v1 = client.BatchV1Api()

# Load the manifest YAML file
yaml_file = "laser-worker-highcpu-deploy-manifests.yaml"

with open(yaml_file) as f:
    manifest = yaml.safe_load(f)

env_vars = {
    "STUDY_NAME": "laser_polio_calib_fixed",
    "NUM_TRIALS": "50",
    "STORAGE_URL": "mysql+pymysql://optuna:superSecretPassword@optuna-mysql:3306/optunaDatabase",
}


def replace_env_vars(obj):
    """Recursively replace placeholders with environment variables."""
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

# Apply the Job
namespace = "default"  # Change if using a different namespace

try:
    response = batch_v1.create_namespaced_job(namespace=namespace, body=manifest)
    print(f"Job {response.metadata.name} created successfully.")
except client.ApiException as e:
    print(f"Error applying the job: {e}")
