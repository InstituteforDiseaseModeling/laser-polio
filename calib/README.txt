To run locally on Windows PowerShell:
$env:STORAGE_URL="sqlite:///optuna.db"; python calib/calibrate.py --num-trials=10


To build docker image, run docker build command from main directory.


# To calibrate on AKS...
1. You'll need to create a file called calib/cloud/local_storage.yaml that specifies the storage_url of the Optuna database. For security reasons, this file has been added to .gitignore.
2. The formatting is as follow:
`storage_url: "STORAGE_URL"`
3. See the docs for instructions on how to get the storage url.
