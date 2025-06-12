import subprocess
import time
import os
import sys
import uuid
import argparse
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def run_command(command):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.stderr.strip()}")
        sys.exit(1)


def download_data(config_file, pod_name, namespace, remote_path, local_path):
    """Download data from the pod to the local directory."""
    print(f"Downloading data from pod '{pod_name}:{remote_path}' to local directory '{local_path}'...")
    if config_file:
        command = f"kubectl --kubeconfig='{config_file}' cp {namespace}/{pod_name}:{remote_path} {local_path}"
    else:
        command = f"kubectl cp {namespace}/{pod_name}:{remote_path} {local_path}"
    print(f"Running command: {command}")
    run_command(command)


def upload_data(config_file, pod_name, namespace, local_path, remote_path):
    """Upload data from the local directory to the pod."""
    print(f"Uploading data from local directory '{local_path}' to pod '{pod_name}:{remote_path}'...")
    if config_file:
        command = f"kubectl --kubeconfig='{config_file}' cp {local_path} {namespace}/{pod_name}:{remote_path}"
    else:
        command = f"kubectl cp {local_path} {namespace}/{pod_name}:{remote_path}"
    print(f"Running command: {command}")
    run_command(command)


def open_shell(config_file, pod_name, namespace):
    """Open a shell session in the pod."""
    print(f"Opening shell session in pod '{pod_name}'...")
    if config_file:
        command = f"kubectl --kubeconfig='{config_file}' exec -it {pod_name} -n {namespace} -- /bin/bash"
    else:
        command = f"kubectl exec -it {pod_name} -n {namespace} -- /bin/bash"
    os.system(command)  # Use os.system to allow interactive shell


def create_pod(pod_name, namespace, data_dir):
    """Create a pod that sleeps forever using the Kubernetes Python client."""
    print(f"Creating pod '{pod_name}'...")

    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name=pod_name, namespace=namespace),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="aks2local-container",
                    image="registry4idm.azurecr.io/nfstest:1.1",
                    command=["sleep", "infinity"],
                    volume_mounts=[
                        client.V1VolumeMount(
                            name="shared-data",
                            mount_path=data_dir
                        )])
            ],
            restart_policy="Never",
            image_pull_secrets=[client.V1LocalObjectReference(name="registry4idm")],
            volumes=[
                client.V1Volume(
                    name="shared-data",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name="laser-stg-pvc"
                    ))
            ]
        )
    )

    # Create the pod
    api_instance = client.CoreV1Api()
    api_instance.create_namespaced_pod(namespace=namespace, body=pod)
    print(f"Pod '{pod_name}' created successfully.")


def wait_for_pod_running(pod_name, namespace):
    """Wait until the pod is in the Running state using the Kubernetes Python client."""
    print(f"Waiting for pod '{pod_name}' to be in Running state...")
    api_instance = client.CoreV1Api()

    while True:
        try:
            pod = api_instance.read_namespaced_pod(name=pod_name, namespace=namespace)
            if pod.status.phase == "Running":
                print(f"Pod '{pod_name}' is now Running.")
                break
        except ApiException as e:
            if e.status != 404:
                print(f"Error while checking pod status: {e}")
                sys.exit(1)
        time.sleep(2)


def delete_pod(pod_name, namespace):
    """Delete the pod using the Kubernetes Python client."""
    print(f"Deleting pod '{pod_name}'...")
    api_instance = client.CoreV1Api()

    try:
        api_instance.delete_namespaced_pod(name=pod_name, namespace=namespace)
        print(f"Pod '{pod_name}' deleted successfully.")
    except ApiException as e:
        print(f"Error while deleting pod: {e}")
        sys.exit(1)


def validate_paths(action, local_dir, remote_dir, shared_data_dir):
    if action in ["download", "upload"]:
        if not local_dir or not remote_dir:
            print(f"Both --local-dir and --remote-dir arguments are required for {action}.")
            sys.exit(1)

        if not os.path.isabs(local_dir):
            local_dir = os.path.abspath(local_dir)

        if action == "download":
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
        elif action == "upload":
            if not os.path.exists(local_dir):
                print(f"Local directory '{local_dir}' does not exist for upload.")
                sys.exit(1)

        if not os.path.isabs(remote_dir):
            print(f"Remote directory '{remote_dir}' must be an absolute path.")
            sys.exit(1)

        if not remote_dir.startswith(shared_data_dir):
            print(f"Remote directory '{remote_dir}' must start with '{shared_data_dir}'.")
            sys.exit(1)


def verify_kubectl(verbose=False):
    """Verify that kubectl is installed and functional."""
    print("Verifying kubectl installation...")
    try:
        result = subprocess.run(["kubectl", "version", "--client"], check=True, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        print("Kubectl is installed and functional.")
        if verbose:
            print(f"Kubectl version details:\n{result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Kubectl is not installed or not functional. Details: {e.stderr.strip()}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Utility for Kubernetes.")
    parser.add_argument("--action", required=True, help="Action to perform: download, upload, or shell.",
                        choices=["download", "upload", "shell"])
    parser.add_argument("--local-dir", help="Path to the local directory for data transfer.")
    parser.add_argument("--remote-dir", help="Path to the remote directory in the pod.")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace (default: 'default').")
    parser.add_argument("--kube-config", help="Path to the kube config file (optional).")
    parser.add_argument("--shared-data-dir", default="/shared",
                        help="Path to the shared data directory in the pod (default: '/shared').")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    verify_kubectl(verbose=args.verbose)

    # Load Kubernetes configuration
    if args.kube_config:
        config.load_kube_config(config_file=args.kube_config)
    else:
        config.load_kube_config()

    print()
    unique_id = str(uuid.uuid4())[:8]  # Generate a short unique identifier
    pod_name = f"kubeutil-pod-{unique_id}"
    namespace = args.namespace
    shared_data_dir = args.shared_data_dir
    validate_paths(args.action, args.local_dir, args.remote_dir, shared_data_dir)

    if args.action == "upload":
        create_pod(pod_name, namespace, shared_data_dir)
        wait_for_pod_running(pod_name, namespace)
        upload_data(args.kube_config, pod_name, namespace, args.local_dir, args.remote_dir)
        delete_pod(pod_name, namespace)
        print("Data upload complete and pod deleted.")

    elif args.action == "download":
        create_pod(pod_name, namespace, shared_data_dir)
        wait_for_pod_running(pod_name, namespace)
        download_data(args.kube_config, pod_name, namespace, args.remote_dir, args.local_dir)
        delete_pod(pod_name, namespace)
        print("Data download complete and pod deleted.")

    elif args.action == "shell":
        create_pod(pod_name, namespace, shared_data_dir)
        wait_for_pod_running(pod_name, namespace)
        open_shell(args.kube_config, pod_name, namespace)
        delete_pod(pod_name, namespace)
        print("Shell session complete and pod deleted.")


if __name__ == "__main__":
    main()
