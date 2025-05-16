import cloud_calib_config as cfg
from kubernetes import client
from kubernetes import config

# Load kubeconfig
config.load_kube_config()
core_v1 = client.CoreV1Api()

# Define the container
container = client.V1Container(
    name=cfg.job_name,
    image=cfg.image,
    image_pull_policy="Always",
    command=["sleep", "infinity"],  # Keeps the pod alive
    env=[client.V1EnvVar(name="NUMBA_NUM_THREADS", value="8")],
    env_from=[client.V1EnvFromSource(secret_ref=client.V1SecretEnvSource(name="mysql-secrets"))],
    resources=client.V1ResourceRequirements(requests={"cpu": "2"}, limits={"cpu": "2"}),
    tty=True,
    stdin=True,
)

# Pod spec
pod_spec = client.V1PodSpec(
    containers=[container],
    restart_policy="Never",
    image_pull_secrets=[client.V1LocalObjectReference(name="idmodregcred3")],
    node_selector={"nodepool": "highcpu"},
    tolerations=[
    client.V1Toleration(
	    key="nodepool",
	    operator="Equal",
	    value="highcpu",
	    effect="NoSchedule"
	    )
    ]
)

# Pod object
pod = client.V1Pod(
    api_version="v1",
    kind="Pod",
    metadata=client.V1ObjectMeta(name=f"{cfg.job_name}-debug"),
    spec=pod_spec
)

# Create the pod
try:
    response = core_v1.create_namespaced_pod(namespace=cfg.namespace, body=pod)
    print(f"✅ Pod {response.metadata.name} created successfully.")
    print(f"👉 Run: kubectl exec -it {response.metadata.name} -n {cfg.namespace} -- bash")
except client.exceptions.ApiException as e:
    print(f"❌ Error creating the pod: {e}")
