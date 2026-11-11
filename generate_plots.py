
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "results" / "assignment3" / "parallel_results.csv"
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Read CSV and extract one summary per configuration
with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

groups = {}

for row in rows:
    key = (
        int(row["size"]),
        row["method"],
        int(row["requested_threads"])
    )
    groups.setdefault(key, []).append(row)

sizes = sorted({key[0] for key in groups})

for size in sizes:
    sequential_key = (size, "sequential_ikj", 1)

    if sequential_key not in groups:
        raise ValueError(f"Missing sequential results for size {size}")

    seq_rows = groups[sequential_key]

    if len(seq_rows) != 7:
        raise ValueError(f"Expected 7 sequential repetitions for size {size}")

    sequential_time = float(seq_rows[0]["median_seconds"])

    thread_counts = []
    parallel_times = []
    time_min_errors = []
    time_max_errors = []
    speedups = []
    speedup_min_errors = []
    speedup_max_errors = []
    efficiencies = []
    efficiency_min_errors = []
    efficiency_max_errors = []

    for threads in [1, 2, 4, 6, 8, 10, 12]:
        key = (size, "parallel_ikj", threads)

        if key not in groups:
            raise ValueError(f"Missing results: size={size}, threads={threads}")

        config_rows = groups[key]

        if len(config_rows) != 7:
            raise ValueError(
                f"Expected 7 repetitions for size={size}, threads={threads}"
            )

        median = float(config_rows[0]["median_seconds"])
        minimum = float(config_rows[0]["min_seconds"])
        maximum = float(config_rows[0]["max_seconds"])

        speedup = sequential_time / median
        efficiency = speedup / threads * 100

        thread_counts.append(threads)
        parallel_times.append(median)

        # Error ranges based on observed minimum and maximum times
        time_min_errors.append(median - minimum)
        time_max_errors.append(maximum - median)

        speedups.append(speedup)
        speedup_min_errors.append(speedup - sequential_time / maximum)
        speedup_max_errors.append(sequential_time / minimum - speedup)

        efficiencies.append(efficiency)
        efficiency_min_errors.append(
            (speedup - sequential_time / maximum) / threads
        )
        efficiency_max_errors.append(
            (sequential_time / minimum - speedup) / threads
        )

    # 1. Execution time
    plt.figure(figsize=(9, 5.5))
    plt.axhline(
        sequential_time,
        linestyle="--",
        label=f"Sequential ({sequential_time:.3f} s)"
    )
    plt.errorbar(
        thread_counts,
        parallel_times,
        yerr=[time_min_errors, time_max_errors],
        marker="o",
        capsize=4,
        label="Parallel IKJ"
    )
    plt.xlabel("Number of threads")
    plt.ylabel("Median execution time (seconds)")
    plt.title(f"Execution time - Matrix size {size} x {size}")
    plt.xticks(thread_counts)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"execution_time_{size}.png", dpi=300)
    plt.close()

    # 2. Speedup
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        thread_counts,
        thread_counts,
        linestyle="--",
        label="Ideal speedup"
    )
    plt.errorbar(
        thread_counts,
        speedups,
        yerr=[speedup_min_errors, speedup_max_errors],
        marker="o",
        capsize=4,
        label="Measured speedup"
    )
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    plt.title(f"Parallel speedup - Matrix size {size} x {size}")
    plt.xticks(thread_counts)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"speedup_{size}.png", dpi=300)
    plt.close()

    # 3. Parallel efficiency
    plt.figure(figsize=(9, 5.5))
    plt.axhline(100, linestyle="--", label="Ideal efficiency (100%)")
    plt.errorbar(
        thread_counts,
        efficiencies,
        yerr=[efficiency_min_errors, efficiency_max_errors],
        marker="o",
        capsize=4,
        label="Measured efficiency"
    )
    plt.xlabel("Number of threads")
    plt.ylabel("Parallel efficiency (%)")
    plt.title(f"Parallel efficiency - Matrix size {size} x {size}")
    plt.xticks(thread_counts)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"efficiency_{size}.png", dpi=300)
    plt.close()

    print(f"Generated plots for matrix size {size} x {size}")

print(f"\nAll plots saved in: {PLOTS_DIR}")
