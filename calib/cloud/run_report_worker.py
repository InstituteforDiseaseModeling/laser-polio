import cloud_calib_config as cfg
from kubernetes import client
from kubernetes import config

# Load kubeconfig
config.load_kube_config(config_file="~/.kube/config")
batch_v1 = client.BatchV1Api()

# === Define the container for report job ===
container = client.V1Container(
    name="report-job",
    image=cfg.image,  # Make sure the image includes run_report_on_aks.py
    image_pull_policy="Always",
    command=["python3", "run_report_on_aks.py"],
    env_from=[client.V1EnvFromSource(secret_ref=client.V1SecretEnvSource(name="mysql-secrets"))],
    resources=client.V1ResourceRequirements(requests={"memory": "16Gi"}),
)

# === Define pod spec ===
template = client.V1PodTemplateSpec(
    spec=client.V1PodSpec(
        containers=[container],
        restart_policy="OnFailure",
        image_pull_secrets=[client.V1LocalObjectReference(name="idmodregcred3")],
        node_selector={"nodepool": "highcpu"},
        tolerations=[client.V1Toleration(key="nodepool", operator="Equal", value="highcpu", effect="NoSchedule")],
    )
)

# === Define job spec ===
job_spec = client.V1JobSpec(template=template, backoff_limit=2, ttl_seconds_after_finished=1200)

# === Create the job object ===
job = client.V1Job(
    api_version="batch/v1",
    kind="Job",
    metadata=client.V1ObjectMeta(name="report-job"),
    spec=job_spec,
)

# === Submit the job ===
try:
    response = batch_v1.create_namespaced_job(namespace=cfg.namespace, body=job)
    print(f"✅ Report job {response.metadata.name} created successfully.")
except client.exceptions.ApiException as e:
    print(f"❌ Failed to create report job: {e}")
