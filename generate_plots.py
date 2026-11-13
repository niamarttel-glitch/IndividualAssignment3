
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import statistics

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "results" / "assignment3" / "parallel_results_limited.csv"
OUTPUT_DIR = BASE_DIR / "plots" / "limited"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

groups = defaultdict(list)

with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    for row in reader:
        key = (int(row["size"]), row["method"], int(row["requested_threads"]))
        groups[key].append(float(row["time_seconds"]))

sizes = sorted({key[0] for key in groups})
thread_counts = [1, 2, 4, 6, 8, 10, 12]

for size in sizes:
    seq_key = (size, "sequential_ikj", 1)
    if seq_key not in groups or len(groups[seq_key]) != 7:
        raise ValueError(f"Invalid sequential measurements for size {size}")

    sequential_median = statistics.median(groups[seq_key])

    times = []
    speedups = []
    efficiencies = []
    time_errors_low = []
    time_errors_high = []

    for threads in thread_counts:
        key = (size, "parallel_ikj", threads)
        if key not in groups or len(groups[key]) != 7:
            raise ValueError(f"Invalid measurements for {key}")

        values = groups[key]
        median = statistics.median(values)
        minimum = min(values)
        maximum = max(values)

        times.append(median)
        speedups.append(sequential_median / median)
        efficiencies.append((sequential_median / median) / threads)
        time_errors_low.append(median - minimum)
        time_errors_high.append(maximum - median)

    # Execution time
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        thread_counts,
        times,
        yerr=[time_errors_low, time_errors_high],
        marker="o",
        capsize=4
    )
    plt.axhline(
        sequential_median,
        linestyle="--",
        label=f"Sequential baseline ({sequential_median:.3f} s)"
    )
    plt.xlabel("Requested OpenMP threads")
    plt.ylabel("Execution time (seconds)")
    plt.title(f"Execution Time - Matrix Size {size}x{size}")
    plt.xticks(thread_counts)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"execution_time_{size}.png", dpi=200)
    plt.close()

    # Speedup
    plt.figure(figsize=(8, 5))
    plt.plot(thread_counts, speedups, marker="o")
    plt.axhline(1.0, linestyle="--", label="Sequential performance")
    plt.xlabel("Requested OpenMP threads")
    plt.ylabel("Speedup")
    plt.title(f"Speedup - Matrix Size {size}x{size}")
    plt.xticks(thread_counts)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"speedup_{size}.png", dpi=200)
    plt.close()

    # Parallel efficiency
    plt.figure(figsize=(8, 5))
    plt.plot(thread_counts, [value * 100 for value in efficiencies], marker="o")
    plt.axhline(100.0, linestyle="--", label="100% efficiency")
    plt.xlabel("Requested OpenMP threads")
    plt.ylabel("Parallel efficiency (%)")
    plt.title(f"Parallel Efficiency - Matrix Size {size}x{size}")
    plt.xticks(thread_counts)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"efficiency_{size}.png", dpi=200)
    plt.close()

    print(f"Generated plots for size {size}")

print(f"All plots saved in: {OUTPUT_DIR}")
