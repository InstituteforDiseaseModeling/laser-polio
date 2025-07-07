import sys
from pathlib import Path

import cloud_calib_config as cfg
import sciris as sc
from kubernetes import client
from kubernetes import config

sys.path.append(str(Path(__file__).resolve().parent.parent))
from get_lp_module_versions import check_version_match

# Compare the version of laser_polio in the Docker image with the version in the GitHub repository
check_version_match(
    repo="InstituteforDiseaseModeling/laser-polio",
    image_name="idm-docker-staging.packages.idmod.org/laser/laser-polio:latest",
    container_path="/app/laser_polio_deps.txt",
)

# Constants for Kubernetes configuration
PERSISTENT_VOLUME_CLAIM_NAME = "laser-stg-pvc"
SHARED_DIR = "/shared"

# Load kubeconfig
config.load_kube_config(config_file="~/.kube/config")  # default = "~/.kube/config"
batch_v1 = client.BatchV1Api()

# Define the container
container = client.V1Container(
    name=cfg.job_name,
    image=cfg.image,
    image_pull_policy="Always",
    command=[
        "python3",
        "calib/calibrate.py",
        "--study-name",
        cfg.study_name,
        "--n-trials",
        str(cfg.n_trials),
        "--model-config",
        cfg.model_config,
        "--calib-config",
        cfg.calib_config,
        "--fit-function",
        cfg.fit_function,
    ],
    # env=[client.V1EnvVar(name="NUMBA_NUM_THREADS", value="4")],
    env_from=[client.V1EnvFromSource(secret_ref=client.V1SecretEnvSource(name="mysql-secrets"))],
    # resources=client.V1ResourceRequirements(requests={"cpu": "6"}, limits={"cpu": "7"}),
    resources=client.V1ResourceRequirements(requests={"memory": "25Gi"}),
    volume_mounts=[client.V1VolumeMount(name="shared-data", mount_path=SHARED_DIR)],
)

# Pod spec
template = client.V1PodTemplateSpec(
    spec=client.V1PodSpec(
        containers=[container],
        restart_policy="OnFailure",
        image_pull_secrets=[client.V1LocalObjectReference(name="idmodregcred3")],
        node_selector={"nodepool": "highcpu"},
        tolerations=[client.V1Toleration(key="nodepool", operator="Equal", value="highcpu", effect="NoSchedule")],
        volumes=[
            client.V1Volume(
                name="shared-data",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=PERSISTENT_VOLUME_CLAIM_NAME),
            )
        ],
    )
)

# Job spec
job_spec = client.V1JobSpec(
    template=template,
    parallelism=cfg.parallelism,
    completions=cfg.completions,
    ttl_seconds_after_finished=120,
    backoff_limit=1000,
)

# Job object
job = client.V1Job(api_version="batch/v1", kind="Job", metadata=client.V1ObjectMeta(name=cfg.job_name), spec=job_spec)

# Apply the job
try:
    response = batch_v1.create_namespaced_job(namespace=cfg.namespace, body=job)
    sc.printgreen(f"✅ Job {response.metadata.name} created successfully.")
except client.exceptions.ApiException as e:
    sc.printred(f"❌ Error applying the job: {e}")
