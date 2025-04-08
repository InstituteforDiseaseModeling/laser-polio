import os


def get_storage():
    if os.getenv("STORAGE_URL"):
        storage_url = os.getenv("STORAGE_URL")
    else:
        # Construct the storage URL from environment variables
        # storage_url = "mysql+pymysql://{}:{}@optuna-mysql:3306/{}".format(
        storage_url = "mysql://{}:{}@mysql:3306/{}".format(
            os.getenv("MYSQL_USER", "root"), os.getenv("MYSQL_PASSWORD", ""), os.getenv("MYSQL_DB", "optuna_db")
        )

    print(f"storage_url={storage_url}")
    return storage_url
