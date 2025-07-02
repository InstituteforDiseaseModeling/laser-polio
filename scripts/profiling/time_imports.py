"""
Script to measure and plot import times for packages used in laser_polio.
Uses subprocess to ensure each import is timed independently in a fresh Python process.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt


def time_import_subprocess(module_name, from_module=None):
    """
    Time how long it takes to import a module in a fresh Python subprocess.

    Args:
        module_name (str): Name of the module to import
        from_module (str, optional): If importing from a specific module

    Returns:
        float: Import time in milliseconds
    """
    # Create the import statement
    if from_module:
        import_statement = f"from {from_module} import {module_name}"
    else:
        import_statement = f"import {module_name}"

    # Create a temporary Python script to run the import
    script_content = f"""
import time
start_time = time.perf_counter()
try:
    {import_statement}
    end_time = time.perf_counter()
    print((end_time - start_time) * 1000)  # Convert to milliseconds
except ImportError as e:
    print("ERROR:", e)
"""

    # Write to temporary file and execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        # Run the subprocess
        result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=True,
            text=True,
            timeout=120,  # 30 second timeout
        )

        if result.returncode == 0 and not result.stdout.startswith("ERROR"):
            return float(result.stdout.strip())
        else:
            print(f"Failed to import {module_name}: {result.stdout.strip()}")
            return 0

    except subprocess.TimeoutExpired:
        print(f"Timeout importing {module_name}")
        return 0
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
        return 0
    finally:
        # Clean up temporary file
        try:
            Path.unlink(temp_script)
        except OSError:
            pass


def time_import_in_process(module_name, from_module=None):
    """
    Alternative method: Time imports by manipulating sys.modules more aggressively.
    This method attempts to remove all related modules from cache.
    """
    # Find all modules that might be related
    modules_to_remove = []
    target_module = from_module if from_module else module_name

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith(target_module):
            modules_to_remove.append(mod_name)

    # Remove modules from cache
    for mod_name in modules_to_remove:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    # Now time the import
    start_time = time.perf_counter()
    try:
        if from_module:
            module = __import__(from_module, fromlist=[module_name])
            getattr(module, module_name)
        else:
            __import__(module_name)

        end_time = time.perf_counter()
        return (end_time - start_time) * 1000  # Convert to milliseconds

    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        return 0


def main():
    """Main function to time imports and create plot."""

    # List of modules to test (based on laser_polio/utils.py imports)
    modules_to_test = [
        ("alive_progress", None),
        ("laser_core", None),
        ("laser_polio", None),
        ("calendar", None),
        ("csv", None),
        ("datetime", None),
        ("json", None),
        ("os", None),
        ("defaultdict", "collections"),
        ("timedelta", "datetime"),
        ("perf_counter_ns", "time"),
        ("ZoneInfo", "zoneinfo"),
        ("matplotlib.colors", None),
        ("matplotlib.pyplot", None),
        ("numba", None),
        ("numpy", None),
        ("pandas", None),
        ("sciris", None),
    ]

    print("Measuring import times using subprocess (fresh Python process for each import)...")
    print("=" * 70)

    import_times_subprocess = {}

    for module_name, from_module in modules_to_test:
        display_name = f"{from_module}.{module_name}" if from_module else module_name
        print(f"Testing {display_name}...", end=" ", flush=True)

        import_time = time_import_subprocess(module_name, from_module)

        # Display name for plotting
        display_name = f"{from_module}.{module_name}" if from_module else module_name
        import_times_subprocess[display_name] = import_time

        print(f"{import_time:8.2f} ms")

    print("=" * 70)
    print(f"Total import time (subprocess): {sum(import_times_subprocess.values()):.2f} ms")

    # Also test with in-process method for comparison
    print("\nMeasuring import times using aggressive cache clearing (same process)...")
    print("=" * 70)

    import_times_inprocess = {}

    for module_name, from_module in modules_to_test:
        import_time = time_import_in_process(module_name, from_module)
        display_name = f"{from_module}.{module_name}" if from_module else module_name
        import_times_inprocess[display_name] = import_time
        print(f"{display_name:<25}: {import_time:8.2f} ms")

    print("=" * 70)
    print(f"Total import time (in-process): {sum(import_times_inprocess.values()):.2f} ms")

    # Create comparison plots
    create_comparison_plot(import_times_subprocess, import_times_inprocess)


def create_comparison_plot(subprocess_times, inprocess_times):
    """Create a comparison plot of both timing methods."""

    # Get common modules
    common_modules = set(subprocess_times.keys()) & set(inprocess_times.keys())
    modules = sorted(common_modules, key=lambda x: subprocess_times[x], reverse=True)

    subprocess_values = [subprocess_times[mod] for mod in modules]
    inprocess_values = [inprocess_times[mod] for mod in modules]

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Subplot 1: Subprocess times
    bars1 = ax1.barh(range(len(modules)), subprocess_values, color="steelblue", alpha=0.7)
    ax1.set_yticks(range(len(modules)))
    ax1.set_yticklabels(modules)
    ax1.set_xlabel("Import Time (milliseconds)")
    ax1.set_title("Import Times - Fresh Process (Subprocess)")
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (_bar, time_val) in enumerate(zip(bars1, subprocess_values, strict=False)):
        if time_val > 0:
            ax1.text(time_val + max(subprocess_values) * 0.01, i, f"{time_val:.1f} ms", va="center", fontsize=8)

    # Subplot 2: In-process times
    bars2 = ax2.barh(range(len(modules)), inprocess_values, color="coral", alpha=0.7)
    ax2.set_yticks(range(len(modules)))
    ax2.set_yticklabels(modules)
    ax2.set_xlabel("Import Time (milliseconds)")
    ax2.set_title("Import Times - Same Process (Cache Cleared)")
    ax2.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (_bar, time_val) in enumerate(zip(bars2, inprocess_values, strict=False)):
        if time_val > 0:
            ax2.text(time_val + max(inprocess_values) * 0.01, i, f"{time_val:.1f} ms", va="center", fontsize=8)

    plt.tight_layout()

    # Save the plot
    plt.savefig("import_times_comparison.png", dpi=300, bbox_inches="tight")
    print("\nComparison plot saved as 'import_times_comparison.png'")

    # Also create a single plot with the subprocess times (most accurate)
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(modules)), subprocess_values, color="steelblue", alpha=0.7)

    plt.yticks(range(len(modules)), modules)
    plt.xlabel("Import Time (milliseconds)")
    plt.title("Package Import Times - Fresh Python Process (Most Accurate)")
    plt.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (_bar, time_val) in enumerate(zip(bars, subprocess_values, strict=False)):
        if time_val > 0:
            plt.text(time_val + max(subprocess_values) * 0.01, i, f"{time_val:.1f} ms", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("import_times_accurate.png", dpi=300, bbox_inches="tight")
    print("Accurate import times plot saved as 'import_times_accurate.png'")

    plt.show()


if __name__ == "__main__":
    main()
