import sys
from pathlib import Path

from click.testing import CliRunner

sys.path.append("calib")
from calibrate import cli


def run_dry_run_with_args(*args):
    runner = CliRunner()
    return runner.invoke(cli, [*args, "--dry-run", "True"])


def test_study_name_override():
    study_name = "test123"
    result = run_dry_run_with_args("--study-name", study_name)

    assert result.exit_code == 0, f"Expected CLI to exit cleanly with code 0, but got {result.exit_code}.\nOutput:\n{result.output}"

    # 1. Check study name string
    expected_sn = f"study_name: {study_name}"
    assert expected_sn in result.output, f"Expected output to contain: '{expected_sn}'\nActual output:\n{result.output}"

    # 2. Check results folder path
    expected_results_path_fragment = str(Path("results") / study_name)
    assert expected_results_path_fragment in result.output, (
        f"Expected output to mention results path fragment: '{expected_results_path_fragment}'\nActual output:\n{result.output}"
    )

    # 3. Check presence of actual_data_file line
    assert "actual_data_file:" in result.output, (
        f"Expected 'actual_data_file:' line to be present in CLI output.\nActual output:\n{result.output}"
    )

    # 4. Check actual_data_file path includes expected folder and filename
    expected_data_file_fragment = str(Path(study_name) / "actual_data.csv")
    assert expected_data_file_fragment in result.output, (
        f"Expected actual data file path to include: '{expected_data_file_fragment}'\nActual output:\n{result.output}"
    )


def test_model_config_override():
    result = run_dry_run_with_args("--model-config", "my_model.yaml")
    assert result.exit_code == 0
    assert "model_config:" in result.output
    assert "my_model.yaml" in result.output


def test_calib_config_override():
    calib_filename = "alt_calib.yaml"
    result = run_dry_run_with_args("--calib-config", calib_filename)

    # Check exit code
    assert result.exit_code == 0, f"Expected CLI to exit with code 0, but got {result.exit_code}.\nOutput:\n{result.output}"

    # Check that the 'calib_config' key appears in output
    assert "calib_config:" in result.output, (
        f"Expected 'calib_config:' to appear in CLI output, but it was missing.\nOutput:\n{result.output}"
    )

    # Check that the specified filename appears in the resolved path
    assert calib_filename in result.output, (
        f"Expected the specified calibration file '{calib_filename}' to appear in output.\n"
        f"This ensures the CLI is resolving and echoing the user's input correctly.\n"
        f"Output:\n{result.output}"
    )


def test_fit_function_override():
    result = run_dry_run_with_args("--fit-function", "log_mse")
    assert result.exit_code == 0
    assert "fit_function: log_mse" in result.output


def test_n_replicates_override():
    result = run_dry_run_with_args("--n-replicates", "5")
    assert result.exit_code == 0
    assert "n_replicates: 5" in result.output


def test_n_trials_override():
    result = run_dry_run_with_args("--n-trials", "7")
    assert result.exit_code == 0
    assert "n_trials: 7" in result.output


def test_all_defaults():
    result = run_dry_run_with_args()
    assert result.exit_code == 0
    assert "n_trials: 2" in result.output
    assert "calib_config:" in result.output
    assert "model_config:" in result.output
    assert "fit_function: log_likelihood" in result.output
    assert "actual_data_file:" in result.output
    assert "n_replicates: 1" in result.output
