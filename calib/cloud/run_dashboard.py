import subprocess
import threading
import time
import webbrowser

import yaml


def load_storage_url(yaml_path="calib/cloud/local_storage.yaml"):
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config["storage_url"]


def port_forward():
    print("🔌 Setting up port forwarding...")
    pf = subprocess.Popen(["kubectl", "port-forward", "mysql-0", "3306:3306"])
    return pf


def start_optuna_dashboard(storage_url):
    print("🚀 Starting Optuna dashboard...")
    local_url = storage_url.replace("localhost", "127.0.0.1")
    subprocess.run(
        [
            r"C:\github\laser-polio\.venv\Scripts\python.exe",
            "-c",
            f"import optuna_dashboard; optuna_dashboard.run_server('{local_url}')",
        ]
    )


def main():
    storage_url = load_storage_url()
    pf_process = port_forward()
    # Give port-forward time to connect before starting the dashboard
    time.sleep(3)

    # Open the dashboard in a browser
    threading.Timer(5, lambda: webbrowser.open("http://localhost:8080")).start()

    try:
        start_optuna_dashboard(storage_url)
    finally:
        print("🧹 Cleaning up port forwarding...")
        pf_process.terminate()


if __name__ == "__main__":
    main()
