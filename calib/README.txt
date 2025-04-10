To run locally on Windows PowerShell:
$env:STORAGE_URL="sqlite:///optuna.db"; python calib/calibrate.py --num-trials=10


To build docker image, run docker build command from main directory.
