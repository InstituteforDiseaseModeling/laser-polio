from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sciris as sc

import laser_polio as lp


def run_memory_profiling():
    """
    Run memory profiling comparing init_younger_than=15 vs init_younger_than=100
    """

    ###################################
    ######### SHARED PARAMETERS #######

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

    # Memory monitoring parameters
    monitor_memory = True
    memory_monitor_interval = 1.0  # Sample every 1 second

    ######### END OF SHARED PARS ######
    ###################################

    # Create results directories
    base_results_path = Path("results/memory_profiling")
    base_results_path.mkdir(parents=True, exist_ok=True)

    # Configuration for both runs
    configurations = [
        {
            "name": "init_under_15",
            "init_younger_than": 15,
            "results_path": base_results_path / "init_under_15",
            "description": "Initialize only agents under 15 years old",
        },
        {
            "name": "init_all_ages",
            "init_younger_than": 100,  # Effectively initialize all ages
            "results_path": base_results_path / "init_all_ages",
            "description": "Initialize all age groups",
        },
    ]

    # Store results for comparison
    memory_results = {}
    simulation_results = {}

    for config in configurations:
        sc.printcyan(f"\n{'=' * 60}")
        sc.printcyan(f"Running simulation: {config['name']}")
        sc.printcyan(f"Description: {config['description']}")
        sc.printcyan(f"init_younger_than: {config['init_younger_than']}")
        sc.printcyan(f"{'=' * 60}")

        lp.print_memory(f"Before {config['name']} simulation")

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
                init_younger_than=config["init_younger_than"],
                monitor_memory=monitor_memory,
                memory_monitor_interval=memory_monitor_interval,
                results_path=str(config["results_path"]),
                save_plots=True,
                save_data=True,
                verbose=1,
                seed=1,
                save_init_pop=False,
                plot_pars=True,
            )

            lp.print_memory(f"After {config['name']} simulation")

            # Store memory statistics
            if hasattr(sim, "memory_monitor") and sim.memory_monitor.memory_usage:
                memory_stats = sim.memory_monitor.get_memory_stats()
                memory_results[config["name"]] = {
                    "stats": memory_stats,
                    "timeline": {
                        "timestamps": sim.memory_monitor.timestamps.copy(),
                        "memory_usage": sim.memory_monitor.memory_usage.copy(),
                    },
                    "config": config,
                }

                sc.printgreen(f"\n{config['name']} Memory Summary:")
                print(f"  Peak memory: {memory_stats['peak_mb']:.1f} MB")
                print(f"  Average memory: {memory_stats['avg_mb']:.1f} MB")
                print(f"  Current memory: {memory_stats['current_mb']:.1f} MB")
                print(f"  Total samples: {memory_stats['total_samples']}")

            # Store simulation for later analysis
            simulation_results[config["name"]] = sim

            sc.printgreen(f"✓ Successfully completed {config['name']} simulation")

        except Exception as e:
            sc.printred(f"✗ Error in {config['name']} simulation: {e!s}")
            continue

    # Create comparison plots and summary
    if len(memory_results) == 2:
        create_memory_comparison(memory_results, base_results_path)
        create_summary_report(memory_results, simulation_results, base_results_path)

    sc.printcyan("\n" + "=" * 60)
    sc.printcyan("Memory profiling complete!")
    sc.printcyan(f"Results saved to: {base_results_path}")
    sc.printcyan("=" * 60)

    return memory_results, simulation_results


def create_memory_comparison(memory_results, base_path):
    """Create comparison plots of memory usage between the two configurations"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    colors = ["#3498db", "#e74c3c"]  # Blue for under_15, Red for all_ages

    # Plot 1: Memory usage over time
    for i, (name, data) in enumerate(memory_results.items()):
        timeline = data["timeline"]
        ax1.plot(
            timeline["timestamps"], timeline["memory_usage"], color=colors[i], linewidth=2, label=data["config"]["description"], alpha=0.8
        )

    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Memory Usage (MB)")
    ax1.set_title("Memory Usage Comparison: init_younger_than=15 vs init_younger_than=100")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Memory statistics comparison
    configs = list(memory_results.keys())
    stats_comparison = {
        "Peak (MB)": [memory_results[config]["stats"]["peak_mb"] for config in configs],
        "Average (MB)": [memory_results[config]["stats"]["avg_mb"] for config in configs],
        "Min (MB)": [memory_results[config]["stats"]["min_mb"] for config in configs],
    }

    x = np.arange(len(configs))
    width = 0.25

    for i, (stat_name, values) in enumerate(stats_comparison.items()):
        ax2.bar(x + i * width, values, width, label=stat_name, alpha=0.8)

    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Memory (MB)")
    ax2.set_title("Memory Statistics Comparison")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([memory_results[config]["config"]["description"] for config in configs], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (stat_name, values) in enumerate(stats_comparison.items()):
        for j, v in enumerate(values):
            ax2.text(j + i * width, v + max(values) * 0.01, f"{v:.0f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(base_path / "memory_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Memory comparison plot saved to: {base_path / 'memory_comparison.png'}")


def create_summary_report(memory_results, simulation_results, base_path):
    """Create a summary report of the memory profiling results"""

    report_lines = []
    report_lines.append("# Memory Profiling Report")
    report_lines.append("=" * 50)
    report_lines.append("")

    # Configuration comparison
    report_lines.append("## Configuration Comparison")
    report_lines.append("")

    for name, data in memory_results.items():
        config = data["config"]
        stats = data["stats"]

        report_lines.append(f"### {config['description']}")
        report_lines.append(f"- **Configuration**: {name}")
        report_lines.append(f"- **init_younger_than**: {config['init_younger_than']}")
        report_lines.append(f"- **Peak Memory**: {stats['peak_mb']:.1f} MB")
        report_lines.append(f"- **Average Memory**: {stats['avg_mb']:.1f} MB")
        report_lines.append(f"- **Min Memory**: {stats['min_mb']:.1f} MB")
        report_lines.append(f"- **Current Memory**: {stats['current_mb']:.1f} MB")
        report_lines.append(f"- **Total Samples**: {stats['total_samples']}")
        report_lines.append("")

    # Memory savings calculation
    if len(memory_results) == 2:
        configs = list(memory_results.keys())
        under_15_stats = memory_results["init_under_15"]["stats"]
        all_ages_stats = memory_results["init_all_ages"]["stats"]

        peak_savings = all_ages_stats["peak_mb"] - under_15_stats["peak_mb"]
        avg_savings = all_ages_stats["avg_mb"] - under_15_stats["avg_mb"]
        peak_savings_pct = (peak_savings / all_ages_stats["peak_mb"]) * 100
        avg_savings_pct = (avg_savings / all_ages_stats["avg_mb"]) * 100

        report_lines.append("## Memory Savings Analysis")
        report_lines.append("")
        report_lines.append(f"**Peak Memory Savings**: {peak_savings:.1f} MB ({peak_savings_pct:.1f}% reduction)")
        report_lines.append(f"**Average Memory Savings**: {avg_savings:.1f} MB ({avg_savings_pct:.1f}% reduction)")
        report_lines.append("")

        if peak_savings > 0:
            report_lines.append("✅ **Conclusion**: Using `init_younger_than=15` provides significant memory savings")
        else:
            report_lines.append("❌ **Conclusion**: No memory savings observed with `init_younger_than=15`")
        report_lines.append("")

    # Population information
    report_lines.append("## Population Information")
    report_lines.append("")

    for name, sim in simulation_results.items():
        if hasattr(sim.pars, "n_ppl_init") and hasattr(sim.pars, "unint_older_pop"):
            total_pop = np.sum(sim.pars.n_ppl)
            init_pop = np.sum(sim.pars.n_ppl_init)
            older_pop = np.sum(sim.pars.unint_older_pop)

            report_lines.append(f"### {name}")
            report_lines.append(f"- **Total Population**: {total_pop:,}")
            report_lines.append(f"- **Initialized Population**: {init_pop:,}")
            report_lines.append(f"- **Uninitialized Older Population**: {older_pop:,}")
            report_lines.append(f"- **Fraction Initialized**: {init_pop / total_pop:.1%}")
            report_lines.append("")

    # Write report
    report_path = base_path / "memory_profiling_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Summary report saved to: {report_path}")

    # Also print key findings to console
    if len(memory_results) == 2:
        sc.printcyan("\n" + "=" * 50)
        sc.printcyan("KEY FINDINGS")
        sc.printcyan("=" * 50)
        print(f"Peak Memory Savings: {peak_savings:.1f} MB ({peak_savings_pct:.1f}% reduction)")
        print(f"Average Memory Savings: {avg_savings:.1f} MB ({avg_savings_pct:.1f}% reduction)")

        if peak_savings > 0:
            sc.printgreen("✅ Using init_younger_than=15 provides memory savings!")
        else:
            sc.printyellow("⚠️  No significant memory savings observed")


if __name__ == "__main__":
    memory_results, simulation_results = run_memory_profiling()
