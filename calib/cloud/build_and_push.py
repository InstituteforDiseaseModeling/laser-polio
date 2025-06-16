import subprocess

import sciris as sc


def run_docker_commands():
    image_tag = "idm-docker-staging.packages.idmod.org/laser/laser-polio:latest"
    dockerfile = "calib/Dockerfile"

    build_cmd = ["docker", "build", ".", "-f", dockerfile, "-t", image_tag]
    create_cmd = ["docker", "create", "--name", "temp_laser", image_tag]
    cp_cmd = ["docker", "cp", "temp_laser:/app/laser_polio_deps.txt", "./laser_polio_deps.txt"]
    rm_cmd = ["docker", "rm", "temp_laser"]
    push_cmd = ["docker", "push", image_tag]

    try:
        # Check if Docker is running
        try:
            subprocess.run(["docker", "info"], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            if "The system cannot find the file specified" in str(e):
                sc.printred("\nERROR: Docker Desktop is not running!")
                sc.printred("Please start Docker Desktop and try again.")
                sc.printred("You can start Docker Desktop from the Start menu or system tray.")
            else:
                sc.printred(f"\nERROR: Docker is not accessible: {e!s}")
                sc.printred("Please ensure Docker Desktop is installed and running.")
            return
        except Exception as e:
            sc.printred(f"\nERROR: Unexpected error checking Docker status: {e!s}")
            return

        # Build image
        subprocess.run(build_cmd, check=True)
        print("Docker image built successfully.")

        # Create container and extract file
        subprocess.run(create_cmd, check=True)
        subprocess.run(cp_cmd, check=True)
        subprocess.run(rm_cmd, check=True)
        print("Extracted 'laser_polio_deps.txt' from the image.")

        # Show where it is and grep for 'laser'
        deps_file = "laser_polio_deps.txt"
        # print(f"\nFile '{deps_file}' saved in current directory. Matches for 'laser':\n")
        # subprocess.run(["grep", "laser", deps_file])

        # Python-native search instead of grep
        try:
            with open(deps_file) as f:
                for line_num, line in enumerate(f, 1):
                    if "laser" in line.lower():
                        print(f"{line_num}: {line.strip()}")
        except FileNotFoundError:
            print(f"Error: Could not find {deps_file}")
        except Exception as e:
            print(f"Error reading file: {e}")
        # Push image
        subprocess.run(push_cmd, check=True)
        print("Docker image pushed successfully.")

    except subprocess.CalledProcessError as e:
        sc.printred(f"Error: {e}")


if __name__ == "__main__":
    print("Starting Docker build and push process...")
    run_docker_commands()
