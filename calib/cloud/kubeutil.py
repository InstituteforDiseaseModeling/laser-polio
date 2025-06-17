import argparse
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from kubernetes import client
from kubernetes import config
from kubernetes.client.rest import ApiException

# Constants for actions
DOWNLOAD_ACTION = "download"
UPLOAD_ACTION = "upload"
SHELL_ACTION = "shell"

# Constants for Kubernetes configuration
IMAGE_WITH_TAR_INSTALLED = "registry4idm.azurecr.io/nfstest:1.1"
REGISTRY_AUTH_NAME = "registry4idm"
PERSISTENT_VOLUME_CLAIM_NAME = "laser-stg-pvc"

# Command line defaults
DEFAULT_SHARED_DIR = "/shared"
DEFAULT_NAMESPACE = "default"


def run_kubectl(verbose: bool, kubectl_path: str, *args):
    """Run the kubectl command and return the output."""
    if kubectl_path is None:
        print("Error: Kubectl path is not set. Please ensure kubectl is installed and available in the PATH.")
        sys.exit(1)
    try:
        command = [kubectl_path, *args]
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.stderr.strip()}")
        sys.exit(1)


def download_dir(
    verbose: bool,
    kubectl_path: str,
    config_file: str,
    pod_name: str,
    namespace: str,
    remote_path: str,
    local_path: str,
):
    """Download data from the pod to the local directory."""
    print(f"Downloading data from pod '{pod_name}:{remote_path}' to local directory '{local_path}'...")
    if config_file:
        run_kubectl(verbose, kubectl_path, f"--kubeconfig={config_file}", "cp", f"{namespace}/{pod_name}:{remote_path}", local_path)
    else:
        run_kubectl(verbose, kubectl_path, "cp", f"{namespace}/{pod_name}:{remote_path}", local_path)


def upload_dir(verbose: bool, kubectl_path: str, config_file: str, pod_name: str, namespace: str, local_path: str, remote_path: str):
    """Upload data from the local directory to the pod."""
    print(f"Uploading data from local directory '{local_path}' to pod '{pod_name}:{remote_path}'...")
    if config_file:
        run_kubectl(verbose, kubectl_path, f"--kubeconfig={config_file}", "cp", local_path, f"{namespace}/{pod_name}:{remote_path}")
    else:
        run_kubectl(verbose, kubectl_path, "cp", local_path, f"{namespace}/{pod_name}:{remote_path}")


def open_shell(verbose: bool, kubectl_path: str, config_file: str, pod_name: str, namespace: str):
    """Open a shell session in the pod."""
    print(f"Opening shell session in pod '{pod_name}'...")
    if config_file:
        command = [kubectl_path, f"--kubeconfig={config_file}", "exec", "-it", pod_name, "-n", namespace, "--", "/bin/bash"]
    else:
        command = [kubectl_path, "exec", "-it", pod_name, "-n", namespace, "--", "/bin/bash"]
    subprocess.run(command)  # Use allow interactive shell


def create_pod(verbose: bool, pod_name: str, namespace: str, data_dir: str):
    """Create a pod that sleeps forever using the Kubernetes Python client."""
    if verbose:
        print(f"Creating pod '{pod_name}'...")

    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name=pod_name, namespace=namespace),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="kubeutil-container",
                    image=IMAGE_WITH_TAR_INSTALLED,
                    command=["sleep", "infinity"],
                    volume_mounts=[client.V1VolumeMount(name="shared-data", mount_path=data_dir)],
                )
            ],
            restart_policy="Never",
            image_pull_secrets=[client.V1LocalObjectReference(name=REGISTRY_AUTH_NAME)],
            volumes=[
                client.V1Volume(
                    name="shared-data",
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(claim_name=PERSISTENT_VOLUME_CLAIM_NAME),
                )
            ],
        ),
    )

    # Create the pod
    api_instance = client.CoreV1Api()
    api_instance.create_namespaced_pod(namespace=namespace, body=pod)
    print(f"Pod '{pod_name}' created successfully.")


def wait_for_pod_running(verbose: bool, pod_name: str, namespace: str):
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


def delete_pod(verbose: bool, pod_name: str, namespace: str):
    """Delete the pod using the Kubernetes Python client."""
    if verbose:
        print(f"Deleting pod '{pod_name}'...")
    api_instance = client.CoreV1Api()

    try:
        api_instance.delete_namespaced_pod(name=pod_name, namespace=namespace)
        print(f"Pod '{pod_name}' deleted successfully.")
    except ApiException as e:
        print(f"Error while deleting pod: {e}")
        sys.exit(1)


def validate_paths(action: str, local: str, remote: str, shared: str) -> str:
    if action in [DOWNLOAD_ACTION, UPLOAD_ACTION]:
        if not local or not remote:
            print(f"Both --local-dir and --remote-dir arguments are required for {action}.")
            sys.exit(1)

        if not Path(local).is_absolute():
            local = str(Path(local).resolve())

        if action == DOWNLOAD_ACTION:
            if not os.path.exists(local):
                os.makedirs(local)
        elif action == UPLOAD_ACTION:
            if not os.path.exists(local):
                print(f"Local directory '{local}' does not exist for {UPLOAD_ACTION}.")
                sys.exit(1)

        if not Path(remote).is_absolute():
            print(f"Remote directory '{remote}' must be an absolute path.")
            sys.exit(1)

        if not remote.startswith(shared):
            print(f"Remote directory '{remote}' must start with '{shared}'.")
            sys.exit(1)

    return local


def find_and_verify_cmd(verbose: bool, cmd: str):
    """Verify that kubectl is installed and functional."""
    full_path_to_bin = shutil.which(cmd)
    if full_path_to_bin:
        if verbose:
            print(f"{cmd} is located at: {full_path_to_bin}")
        return full_path_to_bin
    else:
        print(f"Error: {cmd} was not found in PATH.")
        return None


def find_and_verify_kubectl(verbose: bool):
    """Find and verify the kubectl command."""
    kubectl_path = find_and_verify_cmd(verbose, "kubectl")
    if not kubectl_path:
        return None

    print("Verifying kubectl can be run...")
    try:
        output = run_kubectl(verbose, kubectl_path, "version", "--client")
        print("Kubectl is installed and functional.")
        if verbose:
            print(f"Kubectl version details:\n{output}")
        return kubectl_path
    except subprocess.CalledProcessError as e:
        print(f"Error: Executing Kubectl. Details: {e.stderr.strip()}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Utility for Kubernetes.")
    parser.add_argument(
        "--action",
        required=True,
        help=f"Action to perform: {DOWNLOAD_ACTION}, {UPLOAD_ACTION}, or {SHELL_ACTION}.",
        choices=[DOWNLOAD_ACTION, UPLOAD_ACTION, SHELL_ACTION],
    )
    parser.add_argument("--local-dir", help="Path to the local directory for data transfer.")
    parser.add_argument("--remote-dir", help="Path to the remote directory in the pod.")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help=f"Kubernetes namespace (default: '{DEFAULT_NAMESPACE}').")
    parser.add_argument("--config", help="Path to the kube config file (optional).")
    parser.add_argument(
        "--shared", default=DEFAULT_SHARED_DIR, help=f"Path to the shared data directory in the pod (default: '{DEFAULT_SHARED_DIR}')."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    kubectl_path = find_and_verify_kubectl(args.verbose)
    if not kubectl_path:
        sys.exit(1)

    # Load Kubernetes configuration
    if args.config:
        config.load_kube_config(config_file=args.config)
    else:
        config.load_kube_config()

    print()
    unique_id = str(uuid.uuid4())[:8]  # Generate a short unique identifier
    pod_name = f"kubeutil-pod-{unique_id}"
    args.local_dir = validate_paths(args.action, args.local_dir, args.remote_dir, args.shared)
    create_pod(args.verbose, pod_name, args.namespace, args.shared)

    try:
        wait_for_pod_running(args.verbose, pod_name, args.namespace)
        if args.action == UPLOAD_ACTION:
            upload_dir(args.verbose, kubectl_path, args.config, pod_name, args.namespace, args.local_dir, args.remote_dir)
            print("Data upload complete. Deleting pod...")
        elif args.action == DOWNLOAD_ACTION:
            download_dir(args.verbose, kubectl_path, args.config, pod_name, args.namespace, args.remote_dir, args.local_dir)
            print("Data download complete. Deleting pod...")
        elif args.action == SHELL_ACTION:
            open_shell(args.verbose, kubectl_path, args.config, pod_name, args.namespace)
            print("Shell session complete. Deleting pod...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        delete_pod(args.verbose, pod_name, args.namespace)


if __name__ == "__main__":
    main()
