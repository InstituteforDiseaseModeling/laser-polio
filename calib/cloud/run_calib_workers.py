import cloud_calib_config as cfg
from kubernetes import client
from kubernetes import config

# Load kubeconfig
config.load_kube_config()
batch_v1 = client.BatchV1Api()

# Define the container
container = client.V1Container(
    name=cfg.job_name,
    image=cfg.image,
    image_pull_policy="Always",
    command=["python3", "calibrate.py", "--study-name", cfg.study_name, "--num-trials", str(cfg.num_trials)],
    env=[client.V1EnvVar(name="NUMBA_NUM_THREADS", value="8")],
    env_from=[client.V1EnvFromSource(secret_ref=client.V1SecretEnvSource(name="mysql-secrets"))],
    resources=client.V1ResourceRequirements(requests={"cpu": "6"}, limits={"cpu": "8"}),
)

# Pod spec
template = client.V1PodTemplateSpec(
    spec=client.V1PodSpec(
        containers=[container],
        restart_policy="OnFailure",
        image_pull_secrets=[client.V1LocalObjectReference(name="idmodregcred3")],
        node_selector={"agentpool": "general"},
    )
)

# Job spec
job_spec = client.V1JobSpec(template=template, parallelism=cfg.parallelism, completions=cfg.completions, ttl_seconds_after_finished=120)

# Job object
job = client.V1Job(api_version="batch/v1", kind="Job", metadata=client.V1ObjectMeta(name=cfg.job_name), spec=job_spec)

# Apply the job
try:
    response = batch_v1.create_namespaced_job(namespace=cfg.namespace, body=job)
    print(f"✅ Job {response.metadata.name} created successfully.")
except client.exceptions.ApiException as e:
    print(f"❌ Error applying the job: {e}")
