from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import itertools
from laser_polio.utils import get_seasonality

def visualize_seasonality_response():
    """Visualize sensitivity of get_seasonality() to both amplitude and peak day."""

    amplitudes = np.linspace(0.0, 1.0, 6)  # e.g., 0.0 to 1.0 in 6 steps
    peak_days = np.arange(0, 366, 60)      # every 2 months or so
    day_range = np.arange(364)             # simulation days

    fig, axs = plt.subplots(len(amplitudes), len(peak_days), figsize=(18, 12), sharex=True, sharey=True)
    fig.suptitle("Seasonality patterns across amplitudes and peak days", fontsize=16)

    for i, amp in enumerate(amplitudes):
        for j, peak in enumerate(peak_days):
            seasonality = calculate_full_year_seasonality(amp, peak)
            ax = axs[i, j]
            ax.plot(day_range, seasonality, color="blue")
            ax.set_title(f"A={amp:.2f}, Peak={peak}")
            ax.set_ylim(0, 2)
            ax.tick_params(labelsize=6)
            if i == len(amplitudes) - 1:
                ax.set_xlabel("Day of year", fontsize=8)
            if j == 0:
                ax.set_ylabel("Seasonality", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
class MockSim:
    def __init__(self, seasonal_amplitude, seasonal_peak_doy):
        self.pars = {"seasonal_amplitude": seasonal_amplitude, "seasonal_peak_doy": seasonal_peak_doy}
        self.t = 0
        # Create a full year of dates starting from Jan 1, 2024 (leap year)
        start_date = datetime(2024, 1, 1)
        self.datevec = [start_date + timedelta(days=x) for x in range(364)]


def calculate_full_year_seasonality(amplitude, peak_doy):
    """Helper function to calculate seasonality for a full year."""
    sim = MockSim(amplitude, peak_doy)
    seasonality = []
    for day in range(364):  # Using 366 for leap year to cover all cases
        sim.t = day
        seasonality.append(get_seasonality(sim))
    return np.array(seasonality)


def test_seasonality_peaks():
    """Test that seasonality peaks occur on the specified days."""

    # Test cases with different amplitudes and peak days
    test_cases = [
        {"amplitude": 1.0, "peak_doy": 180},  # Mid-year peak
        {"amplitude": 0.5, "peak_doy": 90},  # Spring peak
        {"amplitude": 0.8, "peak_doy": 270},  # Fall peak
        {"amplitude": 0.3, "peak_doy": 0},  # New Year peak
    ]

    for case in test_cases:
        amplitude = case["amplitude"]
        peak_doy = case["peak_doy"]

        # Calculate seasonality for full year
        seasonality = calculate_full_year_seasonality(amplitude, peak_doy)

        # Find the actual peak day
        actual_peak_day = np.argmax(seasonality)

        # Test peak day alignment (allowing for ±1 day tolerance due to discretization)
        assert abs(actual_peak_day - peak_doy) <= 1, (
            f"Peak misaligned for amplitude={amplitude}, peak_doy={peak_doy}. "
            f"Expected peak on day {peak_doy}, but found peak on day {actual_peak_day}"
        )


def test_seasonality_amplitude():
    """Test that seasonality amplitude matches expectations."""

    test_cases = [
        {"amplitude": 1.0, "peak_doy": 180},
        {"amplitude": 0.5, "peak_doy": 90},
        {"amplitude": 0.2, "peak_doy": 270},
    ]

    for case in test_cases:
        amplitude = case["amplitude"]
        peak_doy = case["peak_doy"]

        # Calculate seasonality for full year
        seasonality = calculate_full_year_seasonality(amplitude, peak_doy)

        # Expected range: [1 - amplitude, 1 + amplitude]
        expected_min = 1 - amplitude
        expected_max = 1 + amplitude
        actual_min = np.min(seasonality)
        actual_max = np.max(seasonality)

        # Test amplitude (allowing for small numerical errors)
        tolerance = 1e-10
        assert abs(actual_min - expected_min) < tolerance, (
            f"Minimum value incorrect for amplitude={amplitude}. Expected {expected_min}, got {actual_min}"
        )
        assert abs(actual_max - expected_max) < tolerance, (
            f"Maximum value incorrect for amplitude={amplitude}. Expected {expected_max}, got {actual_max}"
        )


def test_seasonality_symmetry():
    """Test that seasonality pattern is symmetric around its peak."""

    amplitude = 1.0
    peak_doy = 180

    # Calculate seasonality for full year
    seasonality = calculate_full_year_seasonality(amplitude, peak_doy)

    # Get values 30 days before and after the peak
    window = 30
    peak_idx = np.argmax(seasonality)
    before_peak = seasonality[peak_idx - window : peak_idx]
    after_peak = seasonality[peak_idx : peak_idx + window][::-1]  # Reverse for comparison

    # Test symmetry (allowing for small numerical errors)
    np.testing.assert_allclose(before_peak, after_peak, rtol=1e-2, err_msg="Seasonality pattern is not symmetric around peak")


def test_edge_cases():
    """Test edge cases and special values."""

    # Test zero amplitude (should be constant 1.0)
    seasonality = calculate_full_year_seasonality(amplitude=0.0, peak_doy=180)
    np.testing.assert_allclose(seasonality, 1.0, rtol=1e-10, err_msg="Zero amplitude should give constant seasonality of 1.0")

    # Test very small amplitude
    seasonality = calculate_full_year_seasonality(amplitude=1e-6, peak_doy=180)
    assert np.all(seasonality > 0.999), "Very small amplitude should give values very close to 1.0"

    # Test peak_doy at year boundaries
    for peak_doy in [0, 365]:
        seasonality = calculate_full_year_seasonality(amplitude=1.0, peak_doy=peak_doy)
        actual_peak = np.argmax(seasonality)
        assert abs(actual_peak - peak_doy) <= 3, f"Peak misaligned for boundary case peak_doy={peak_doy}"


if __name__ == "__main__":
    test_seasonality_peaks()
    test_seasonality_amplitude()
    test_seasonality_symmetry()
    test_edge_cases()
    # looks like we don't need to put this behind any conditional since if you run with pytest, it doesn't let it show anyway.
    # Just run script "normally" to get this nice visual verification.
    visualize_seasonality_response()
    print("All seasonality tests passed!")
