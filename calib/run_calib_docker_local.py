import re
import subprocess
import uuid
from pathlib import Path

import optuna

# From inside container
STORAGE_URL = "mysql://root@optuna-mysql:3306/optuna_db"
# From outside container
STORAGE_URL2 = "mysql+pymysql://root@127.0.0.1:3306/optuna_db"
IMAGE_NAME = "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest"


def create_study_directory(study_name, model_config, calib_config):
    """Create a study directory and dump the model_config and calib_config."""
    study_dir = Path(study_name)
    study_dir.mkdir(parents=True, exist_ok=True)

    source_paths = [model_config, calib_config]
    dest_paths = [
        study_dir / "model_config.yaml",
        study_dir / "calib_config.yaml",
    ]

    # Create a dictionary mapping container_path → host_path
    files_to_copy = dict(zip(source_paths, dest_paths, strict=False))

    docker_copy_from_image(IMAGE_NAME, files_to_copy, study_dir)

    print(f"✅ Study directory '{study_name}' created with config files")


def docker_copy_from_image(image: str, files_to_copy: dict, output_dir: Path):
    """
    Copy files from a Docker image (not a running container) to the host filesystem.

    Args:
        image (str): Docker image name (e.g., "laser/laser-polio:latest")
        files_to_copy (dict): Mapping from container_path to output_filename (host-relative)
        output_dir (Path): Destination directory on host
    """
    container_name = f"temp_container_{uuid.uuid4().hex[:8]}"
    try:
        subprocess.run(["docker", "create", "--name", container_name, image], check=True)

        for container_path, host_filename in files_to_copy.items():
            dest_path = output_dir / host_filename
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["docker", "cp", f"{container_name}:{container_path}", str(dest_path)], check=True)

    finally:
        subprocess.run(["docker", "rm", container_name], check=False)


def get_default_config_values():
    """Run docker container with --help to retrieve default values for configs."""
    result = subprocess.run(
        ["docker", "run", "--rm", "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest", "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Exception("Docker command failed: " + result.stderr)

    help_output = result.stdout

    def extract_path(flag_name):
        for line in help_output.splitlines():
            if flag_name in line:
                # Look for pattern like: [default: /some/path.yaml]
                match = re.search(r"\[default:\s*(.*?)\]", line)
                if match:
                    return match.group(1)
        return None

    model_config_path = extract_path("--model-config")
    calib_config_path = extract_path("--calib-config")

    return model_config_path, calib_config_path


def get_laser_polio_deps(study_name):
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
        with open(study_name + "/laser_polio_deps.txt", "w") as f:
            f.write(deps_output)
        print("✅ laser_polio_deps.txt retrieved and saved locally.")

    except subprocess.CalledProcessError as e:
        print("❌ Failed to retrieve laser_polio_deps.txt:")
        print(e.stderr)


def run_docker_calibration(study_name, n_trials=2):
    """Run the docker container to perform the calibration with a study."""
    model_config, calib_config = get_default_config_values()

    # Step 1: Save initial config inputs
    create_study_directory(study_name, model_config, calib_config)
    get_laser_polio_deps(study_name)

    # Step 2: Launch container
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name",
        "calib_worker",
        "--network",
        "optuna-network",
        "-e",
        f"STORAGE_URL={STORAGE_URL}",
        "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest",
        "--study-name",
        study_name,
        "--n-trials",
        str(n_trials),
    ]

    result = subprocess.run(docker_command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception("Docker calibration failed: " + result.stderr)

    print(f"✅ Calibration complete for study: {study_name}")


if __name__ == "__main__":
    study_name = "calib_demo_nigeria2"
    run_docker_calibration(study_name, n_trials=1)

    # Step 3: Post-execution study reporting
    from calib.report import plot_optuna
    from calib.report import save_study_results

    storage_url = STORAGE_URL2
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.storage_url = storage_url
    study.study_name = study_name
    save_study_results(study, Path(study_name))
    plot_optuna(study_name, study.storage_url)
