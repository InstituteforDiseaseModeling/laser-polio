import subprocess
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import sciris as sc

import laser_polio as lp


def get_git_branch():
    """Get the current git branch name"""
    try:
        result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True)
        branch_name = result.stdout.strip()
        # Sanitize branch name for use in filenames (replace problematic characters)
        sanitized_name = branch_name.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
        return sanitized_name
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown_branch"


def run_memory_profiling():
    """
    Run memory profiling for a single configuration
    """

    ###################################
    ######### SIMULATION PARAMETERS ###

    regions = [
        "NIGERIA",
    ]
    admin_level = 0
    start_year = 2017
    n_days = 2655
    pop_scale = 1 / 1
    init_region = "BIRINIWA"
    init_prev = 200
    r0 = 10
    migration_method = "radiation"
    radiation_k = 0.5
    max_migr_frac = 1.0
    vx_prob_ri = 0.0
    missed_frac = 0.1
    use_pim_scalars = False
    seed_schedule = [
        {"date": "2017-10-01", "dot_name": "AFRO:NIGERIA:JIGAWA:HADEJIA", "prevalence": 100},
        {"date": "2017-10-01", "dot_name": "AFRO:NIGERIA:JIGAWA:GARKI", "prevalence": 100},
        {"date": "2020-07-01", "dot_name": "AFRO:NIGERIA:ZAMFARA:TALATA_MAFARA", "prevalence": 100},
        {"date": "2020-10-01", "dot_name": "AFRO:NIGERIA:NIGER:SULEJA", "prevalence": 100},
    ]

    # Memory monitoring will be handled manually with our MemoryProfiler class

    ######### END OF SIMULATION PARS ##
    ###################################

    # Get git branch name for configuration
    branch_name = get_git_branch()

    # Create results directories
    base_results_path = Path("results/memory_profiling")
    base_results_path.mkdir(parents=True, exist_ok=True)
    config_results_path = base_results_path / branch_name
    config_results_path.mkdir(parents=True, exist_ok=True)

    sc.printcyan(f"\n{'=' * 60}")
    sc.printcyan("Running memory profiling")
    sc.printcyan(f"Git branch: {branch_name}")
    sc.printcyan(f"Results path: {config_results_path}")
    sc.printcyan(f"{'=' * 60}")

    # Import our memory profiler from the other script

    class SimpleMemoryProfiler:
        def __init__(self, sampling_interval=1.0):
            self.memory_data = []
            self.time_data = []
            self.start_time = None
            self.process = psutil.Process()
            self.sampling_interval = sampling_interval
            self.monitoring = False
            self.monitor_thread = None

        def start_profiling(self):
            self.start_time = time.time()
            self.memory_data = []
            self.time_data = []
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._continuous_monitor, daemon=True)
            self.monitor_thread.start()

        def _continuous_monitor(self):
            while self.monitoring:
                if self.start_time is not None:
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / (1024**2)
                    elapsed_time = time.time() - self.start_time

                    self.memory_data.append(memory_mb)
                    self.time_data.append(elapsed_time)

                time.sleep(self.sampling_interval)

        def stop_profiling(self):
            if self.monitoring:
                self.monitoring = False
                if self.monitor_thread:
                    self.monitor_thread.join()

        def get_stats(self):
            if not self.memory_data:
                return {}
            return {
                "peak_mb": max(self.memory_data),
                "min_mb": min(self.memory_data),
                "avg_mb": sum(self.memory_data) / len(self.memory_data),
                "current_mb": self.memory_data[-1] if self.memory_data else 0,
                "total_samples": len(self.memory_data),
            }

    # Start memory profiling
    profiler = SimpleMemoryProfiler(sampling_interval=1.0)
    profiler.start_profiling()

    lp.print_memory("Before simulation")

    try:
        # Run the simulation
        sim = lp.run_sim(
            regions=regions,
            admin_level=admin_level,
            start_year=start_year,
            n_days=n_days,
            pop_scale=pop_scale,
            init_region=init_region,
            init_prev=init_prev,
            seed_schedule=seed_schedule,
            r0=r0,
            migration_method=migration_method,
            radiation_k=radiation_k,
            max_migr_frac=max_migr_frac,
            vx_prob_ri=vx_prob_ri,
            missed_frac=missed_frac,
            use_pim_scalars=use_pim_scalars,
            results_path=str(config_results_path),
            save_plots=False,
            save_data=False,
            verbose=1,
            seed=1,
            save_init_pop=False,
            plot_pars=False,
        )

        lp.print_memory("After simulation")

        # Stop memory profiling
        profiler.stop_profiling()

        # Store memory statistics and create plots
        memory_stats = profiler.get_stats()
        if memory_stats:
            memory_results = {
                "stats": memory_stats,
                "timeline": {
                    "timestamps": profiler.time_data.copy(),
                    "memory_usage": profiler.memory_data.copy(),
                },
                "branch": branch_name,
            }

            sc.printgreen("\nMemory Summary:")
            print(f"  Peak memory: {memory_stats['peak_mb']:.1f} MB")
            print(f"  Average memory: {memory_stats['avg_mb']:.1f} MB")
            print(f"  Current memory: {memory_stats['current_mb']:.1f} MB")
            print(f"  Total samples: {memory_stats['total_samples']}")

            # Create memory usage plot
            create_memory_plot(memory_results, base_results_path, branch_name)
            create_summary_report(memory_results, sim, base_results_path, branch_name)
        else:
            print("No memory data collected")

        sc.printgreen("✓ Successfully completed simulation")

    except Exception as e:
        sc.printred(f"✗ Error in simulation: {e!s}")
        return None, None

    sc.printcyan("\n" + "=" * 60)
    sc.printcyan("Memory profiling complete!")
    sc.printcyan(f"Results saved to: {base_results_path}")
    sc.printcyan("=" * 60)

    return memory_results, sim


def create_memory_plot(memory_results, base_path, branch_name):
    """Create memory usage plot over time"""

    plt.figure(figsize=(14, 8))

    timeline = memory_results["timeline"]
    stats = memory_results["stats"]

    # Plot memory usage over time
    plt.plot(timeline["timestamps"], timeline["memory_usage"], color="#3498db", linewidth=2, alpha=0.8)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Usage (MB)")
    plt.title(f"Memory Usage Over Time - Branch: {branch_name}")
    plt.grid(True, alpha=0.3)

    # Add peak memory annotation
    peak_memory = stats["peak_mb"]
    peak_idx = timeline["memory_usage"].index(peak_memory)
    peak_time = timeline["timestamps"][peak_idx]

    plt.annotate(
        f"Peak: {peak_memory:.1f} MB",
        xy=(peak_time, peak_memory),
        xytext=(peak_time + max(timeline["timestamps"]) * 0.1, peak_memory),
        arrowprops={"arrowstyle": "->", "color": "red"},
        fontsize=12,
        color="red",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
    )

    # Add text box with statistics
    stats_text = f"""Statistics:
Peak: {stats["peak_mb"]:.1f} MB
Average: {stats["avg_mb"]:.1f} MB
Min: {stats["min_mb"]:.1f} MB
Samples: {stats["total_samples"]}"""

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightblue", "alpha": 0.8},
        fontsize=10,
    )

    plt.tight_layout()

    # Save with branch name in filename
    plot_filename = f"memory_usage_over_time_{branch_name}.png"
    plot_path = base_path / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Memory usage plot saved to: {plot_path}")


def create_summary_report(memory_results, sim, base_path, branch_name):
    """Create a summary report of the memory profiling results"""

    report_lines = []
    report_lines.append("# Memory Profiling Report")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Configuration information
    report_lines.append("## Configuration")
    report_lines.append("")
    report_lines.append(f"- **Git Branch**: {branch_name}")
    report_lines.append(f"- **Peak Memory**: {memory_results['stats']['peak_mb']:.1f} MB")
    report_lines.append(f"- **Average Memory**: {memory_results['stats']['avg_mb']:.1f} MB")
    report_lines.append(f"- **Min Memory**: {memory_results['stats']['min_mb']:.1f} MB")
    report_lines.append(f"- **Current Memory**: {memory_results['stats']['current_mb']:.1f} MB")
    report_lines.append(f"- **Total Samples**: {memory_results['stats']['total_samples']}")
    report_lines.append("")

    # Population information
    if hasattr(sim.pars, "n_ppl"):
        report_lines.append("## Population Information")
        report_lines.append("")

        total_pop = np.sum(sim.pars.n_ppl)
        report_lines.append(f"- **Total Population**: {total_pop:,}")

        if hasattr(sim.pars, "n_ppl_init"):
            init_pop = np.sum(sim.pars.n_ppl_init)
            report_lines.append(f"- **Initialized Population**: {init_pop:,}")
            report_lines.append(f"- **Fraction Initialized**: {init_pop / total_pop:.1%}")

        if hasattr(sim.pars, "unint_older_pop"):
            older_pop = np.sum(sim.pars.unint_older_pop)
            report_lines.append(f"- **Uninitialized Older Population**: {older_pop:,}")

        report_lines.append("")

    # Write report
    report_filename = f"memory_profiling_report_{branch_name}.md"
    report_path = base_path / report_filename
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Summary report saved to: {report_path}")


if __name__ == "__main__":
    memory_results, sim = run_memory_profiling()
