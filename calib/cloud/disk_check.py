# disk_check.py

import time
import uuid

from kubernetes import client
from kubernetes import config


def run_disk_check(namespace="default", pvc_name="laser-stg-pvc", mount_path="/shared", timeout_seconds=30):
    """
    Launch a short-lived pod to check disk usage (df -h) on a mounted PVC.

    Args:
        namespace (str): Kubernetes namespace.
        pvc_name (str): Name of the PersistentVolumeClaim.
        mount_path (str): Mount path to inspect (e.g., "/shared").
        timeout_seconds (int): How long to wait before giving up (in seconds).
    """
    config.load_kube_config()
    core_api = client.CoreV1Api()

    name = f"disk-check-{uuid.uuid4().hex[:5]}"
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[
                client.V1Container(
                    name="disk-checker",
                    image="busybox",
                    command=["sh", "-c", f"echo '[INFO] Disk usage for {mount_path}:'; df -h {mount_path}; sleep 2"],
                    volume_mounts=[client.V1VolumeMount(name="shared-data", mount_path=mount_path)],
                )
            ],
            volumes=[
                client.V1Volume(name="shared-data", persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name))
            ],
        ),
    )

    print(f"[INFO] Launching disk check pod '{name}' in namespace '{namespace}'...")
    try:
        core_api.create_namespaced_pod(namespace=namespace, body=pod)
        for _ in range(timeout_seconds):
            pod_status = core_api.read_namespaced_pod_status(name=name, namespace=namespace)
            if pod_status.status.phase in ("Succeeded", "Failed"):
                break
            time.sleep(1)
        else:
            print("[WARN] Disk check pod did not finish in time.")

        logs = core_api.read_namespaced_pod_log(name=name, namespace=namespace)
        print("[DISK USAGE REPORT]")
        print(logs)

    except Exception as e:
        print(f"[ERROR] Disk check failed: {e}")
    finally:
        try:
            core_api.delete_namespaced_pod(name=name, namespace=namespace, body=client.V1DeleteOptions())
        except Exception:
            pass
