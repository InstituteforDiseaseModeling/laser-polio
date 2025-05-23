import sys

from click.testing import CliRunner

sys.path.append("calib")
from calibrate import cli


def run_dry_run_with_args(*args):
    runner = CliRunner()
    return runner.invoke(cli, [*args, "--dry-run", "True"])


def test_study_name_override():
    result = run_dry_run_with_args("--study-name", "test123")
    assert result.exit_code == 0
    assert "study_name: test123" in result.output
    assert "results/test123" in result.output
    assert "actual_data_file: " in result.output
    assert "test123/actual_data.csv" in result.output


def test_model_config_override():
    result = run_dry_run_with_args("--model-config", "my_model.yaml")
    assert result.exit_code == 0
    assert "model_config:" in result.output
    assert "my_model.yaml" in result.output


def test_calib_config_override():
    result = run_dry_run_with_args("--calib-config", "alt_calib.yaml")
    assert result.exit_code == 0
    assert "calib_config:" in result.output
    assert "alt_calib.yaml" in result.output


def test_fit_function_override():
    result = run_dry_run_with_args("--fit-function", "log_mse")
    assert result.exit_code == 0
    assert "fit_function: log_mse" in result.output


def test_n_replicates_override():
    result = run_dry_run_with_args("--n-replicates", "5")
    assert result.exit_code == 0
    assert "n_replicates: 5" in result.output


def test_num_trials_override():
    result = run_dry_run_with_args("--num-trials", "7")
    assert result.exit_code == 0
    assert "num_trials: 7" in result.output


def test_all_defaults():
    result = run_dry_run_with_args()
    assert result.exit_code == 0
    assert "num_trials: 2" in result.output
    assert "calib_config:" in result.output
    assert "model_config:" in result.output
    assert "fit_function: log_likelihood" in result.output
    assert "actual_data_file:" in result.output
    assert "n_replicates: 1" in result.output
