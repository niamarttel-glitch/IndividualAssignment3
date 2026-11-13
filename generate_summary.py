
import csv
import statistics
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "results" / "assignment3" / "parallel_results_limited.csv"
OUTPUT_FILE = BASE_DIR / "results" / "assignment3" / "summary_results_limited.csv"

groups = defaultdict(list)

with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    for row in reader:
        key = (int(row["size"]), row["method"], int(row["requested_threads"]))
        groups[key].append(float(row["time_seconds"]))

sizes = sorted({key[0] for key in groups})
results = []

for size in sizes:
    seq_key = (size, "sequential_ikj", 1)
    if seq_key not in groups:
        raise ValueError(f"Missing sequential baseline for size {size}")

    seq_times = groups[seq_key]
    if len(seq_times) != 7:
        raise ValueError(f"Expected 7 sequential repetitions for size {size}")

    sequential_median = statistics.median(seq_times)
    candidates = []

    for key, times in groups.items():
        if key[0] != size or key[1] != "parallel_ikj":
            continue
        if len(times) != 7:
            raise ValueError(f"Expected 7 repetitions for {key}")

        threads = key[2]
        median_time = statistics.median(times)
        speedup = sequential_median / median_time
        efficiency = speedup / threads

        candidates.append({
            "size": size,
            "threads": threads,
            "median_seconds": median_time,
            "stddev_seconds": statistics.pstdev(times),
            "speedup": speedup,
            "efficiency": efficiency,
        })

    best = min(candidates, key=lambda item: item["median_seconds"])

    results.append({
        "size": size,
        "sequential_median_seconds": sequential_median,
        "best_threads": best["threads"],
        "best_parallel_median_seconds": best["median_seconds"],
        "speedup": best["speedup"],
        "efficiency": best["efficiency"],
        "parallel_stddev_seconds": best["stddev_seconds"],
    })

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Summary saved to: {OUTPUT_FILE}")
for row in results:
    print(
        f"Size {row['size']}: "
        f"{row['best_threads']} threads, "
        f"{row['best_parallel_median_seconds']:.6f} s, "
        f"speedup {row['speedup']:.3f}x, "
        f"efficiency {row['efficiency'] * 100:.2f}%"
    )
