@echo off
REM Launcher script for Zamfara R0 calibration
python calib\calibrate.py ^
  --study-name calib_demo_zamfara_r0_v2 ^
  --n-trials 2 ^
  --calib-config calib\calib_configs\calib_pars_r0.yaml ^
  --model-config calib\model_configs\config_zamfara.yaml ^
  --sim-path calib\setup_sim.py ^
  --results-path calib\results\calib_demo_zamfara_r0_v2 ^
  --params-file params.json ^
  --actual-data-file examples\calib_demo_zamfara\synthetic_infection_counts_zamfara_r14.csv
