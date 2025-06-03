import io
import tarfile
from pathlib import Path

import docker


def extract_file_from_image(image_name, container_path, local_path):
    client = docker.from_env()

    print(f"Creating temporary container from image: {image_name}")
    container = client.containers.create(image=image_name, name="temp_laser", command="sleep 10")

    try:
        print(f"Extracting file from: {container_path}")
        tar_stream, stat = container.get_archive(container_path)

        file_like = io.BytesIO(b"".join(tar_stream))
        with tarfile.open(fileobj=file_like) as tar:
            member = tar.getmembers()[0]
            extracted_file = tar.extractfile(member)
            if extracted_file:
                with open(local_path, "wb") as f:
                    f.write(extracted_file.read())
                local_path_absolute = Path(local_path).resolve()
                print(f"✅ File extracted: {local_path_absolute}")
            else:
                print("❌ Failed to extract file.")
                return

    finally:
        # print("Cleaning up temporary container...")
        container.remove(force=True)

    # Simulate `grep laser` output
    # print(f"\n🔍 Lines containing 'laser' in {local_path}:")
    with open(local_path) as f:
        for line in f:
            if "laser" in line.lower():
                print(line.rstrip())


# Example usage
extract_file_from_image(
    image_name="idm-docker-staging.packages.idmod.org/laser/laser-polio:latest",
    container_path="/app/laser_polio_deps.txt",
    local_path="./laser_polio_deps.txt",
)
