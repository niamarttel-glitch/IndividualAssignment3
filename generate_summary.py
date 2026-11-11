
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "results" / "assignment3" / "parallel_results.csv"
OUTPUT_FILE = BASE_DIR / "results" / "assignment3" / "summary_results.csv"

with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

groups = {}

for row in rows:
    key = (int(row["size"]), row["method"], int(row["requested_threads"]))
    groups.setdefault(key, []).append(row)

summary = []

for size in sorted({key[0] for key in groups}):
    seq_key = (size, "sequential_ikj", 1)
    seq_rows = groups[seq_key]

    sequential_time = float(seq_rows[0]["median_seconds"])
    best = None

    for threads in [1, 2, 4, 6, 8, 10, 12]:
        key = (size, "parallel_ikj", threads)
        config_rows = groups[key]

        parallel_time = float(config_rows[0]["median_seconds"])
        speedup = sequential_time / parallel_time
        efficiency = speedup / threads * 100

        if best is None or parallel_time < best["parallel_time"]:
            best = {
                "threads": threads,
                "parallel_time": parallel_time,
                "speedup": speedup,
                "efficiency": efficiency,
                "stddev": float(config_rows[0]["stddev_seconds"]),
                "actual_threads": int(config_rows[0]["actual_threads"]),
                "peak_memory": int(config_rows[0]["peak_working_set_bytes"])
            }

    summary.append({
        "size": size,
        "sequential_time_seconds": sequential_time,
        "best_threads": best["threads"],
        "best_parallel_time_seconds": best["parallel_time"],
        "best_speedup": best["speedup"],
        "best_efficiency_percent": best["efficiency"],
        "best_stddev_seconds": best["stddev"],
        "actual_threads": best["actual_threads"],
        "peak_working_set_mib": best["peak_memory"] / (1024 * 1024)
    })

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("SUMMARY RESULTS")
print("-" * 85)

for item in summary:
    print(
        f'{item["size"]:4} x {item["size"]:4} | '
        f'Sequential: {item["sequential_time_seconds"]:.3f} s | '
        f'Best: {item["best_threads"]:2} threads | '
        f'Parallel: {item["best_parallel_time_seconds"]:.3f} s | '
        f'Speedup: {item["best_speedup"]:.3f}x | '
        f'Efficiency: {item["best_efficiency_percent"]:.2f}%'
    )

print(f"\nSummary saved to: {OUTPUT_FILE}")
