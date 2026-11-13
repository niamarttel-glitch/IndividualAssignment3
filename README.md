# Assignment 3 — Parallel Matrix Multiplication

## Project Overview

This project investigates how thread count affects the performance of a shared-memory dense matrix multiplication, and when parallel execution helps.

A sequential C implementation is compared with an OpenMP parallel implementation of the same algorithm. Both use the IKJ loop ordering, row-major storage, double-precision values (`double`) and identical input matrices. An independent IJK implementation is included only for correctness testing.

The study reports execution time, variability, speedup, parallel efficiency and memory consumption for three matrix sizes (512, 1024 and 2048) and seven thread counts (1, 2, 4, 6, 8, 10 and 12).

The complete analysis is in `report/report.pdf`.

## Main Results

Times are medians of 7 repetitions. Speedup is the sequential median divided by the parallel median, and efficiency is speedup divided by the number of threads.

| Matrix size | Sequential (s) | Best threads | Parallel (s) | Speedup | Efficiency |
|---|---:|---:|---:|---:|---:|
| 512 × 512 | 0.045 | 6 | 0.011 | 4.091× | 68.18% |
| 1024 × 1024 | 0.498 | 8 | 0.085 | 5.859× | 73.24% |
| 2048 × 2048 | 4.078 | 12 | 1.001 | 4.074× | 33.95% |

- The best thread count depends on matrix size.
- For 512 × 512 and 1024 × 1024, using 8–12 threads (512) or 10–12 threads (1024) was slower and much more variable than the best configuration.
- For 2048 × 2048, performance kept improving up to 12 threads, with low efficiency.
- The OpenMP version with one thread differs from the sequential version by +2.22%, −7.83% and −0.07% for the three sizes. The −7.83% at 1024 × 1024 is not a real OpenMP gain; see the report for the analysis.
- Peak process working set: 13.00 MiB, 36.99 MiB and 132.96 MiB for 512, 1024 and 2048.

## Project Structure

```text
IA3/
├── src/
│   └── c/
│       ├── parallel.c
│       └── test_openmp.c
├── results/
│   └── assignment3/
│       ├── parallel_results_limited.csv
│       └── summary_results_limited.csv
├── plots/
│   └── limited/
│       ├── execution_time_512.png
│       ├── execution_time_1024.png
│       ├── execution_time_2048.png
│       ├── speedup_512.png
│       ├── speedup_1024.png
│       ├── speedup_2048.png
│       ├── efficiency_512.png
│       ├── efficiency_1024.png
│       └── efficiency_2048.png
├── report/
│   └── report.pdf
├── generate_plots.py
├── generate_summary.py
└── README.md
```

- `src/c/parallel.c`: sequential and OpenMP implementations, IJK reference, correctness tests and benchmark.
- `src/c/test_openmp.c`: OpenMP environment test.
- `results/assignment3/parallel_results_limited.csv`: raw measurements.
- `results/assignment3/summary_results_limited.csv`: best configuration per matrix size.
- `plots/limited/`: execution time, speedup and efficiency plots.
- `report/report.pdf`: final report.

Compiled executables are not included in the repository. They can be generated with the compilation instructions below.

## Hardware and Software

| Component | Specification |
|---|---|
| CPU | 12th Gen Intel Core i5-12450H |
| Physical cores | 8 |
| Logical processors | 12 |
| RAM | 15.73 GB |
| Operating system | Windows |
| Compiler | GCC 16.2.0 (MSYS2 UCRT64) |
| Parallelisation framework | OpenMP |
| Language / numeric type | C / `double` |
| Matrix storage | Row-major |
| Python | 3.14.2 |
| Matplotlib | 3.10.8 |

The processor has a hybrid architecture with performance cores and efficiency cores. No manual processor affinity was configured: thread scheduling was handled by the operating system with the default process affinity.

The largest thread count tested (12) equals the number of logical processors, so no configuration uses more threads than hardware execution contexts (no oversubscription).

## Dependencies

- Windows operating system.
- MSYS2 with the UCRT64 GCC toolchain (GCC 16.2.0) and OpenMP support.
- Python 3.14.2.
- Matplotlib 3.10.8.

Install Matplotlib with:

```powershell
python -m pip install matplotlib==3.10.8
```

The C program uses the Windows Process Status API to measure the process peak working set, so it is linked against the Windows `psapi` library.

## Compilation

Open PowerShell in the project root directory and configure the compiler path:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
```

Compile the program:

```powershell
gcc -O2 -std=c11 -Wall -Wextra -fopenmp .\src\c\parallel.c -o .\src\c\parallel.exe -lpsapi -lm
```

## Running the Benchmarks

Run the program from the project root with the desired square matrix dimension, one size per run:

```powershell
.\src\c\parallel.exe 512
.\src\c\parallel.exe 1024
.\src\c\parallel.exe 2048
```

The folder `results/assignment3/` must exist, because the program writes its CSV file there using a relative path.

Each run performs the following steps:

1. Runs the correctness tests and stops if any of them fails.
2. Checks the theoretical memory requirement against the memory budget.
3. Generates the input matrices from a fixed random seed (`123456789`).
4. Computes the sequential reference result.
5. Benchmarks the sequential baseline, then the OpenMP implementation with 1, 2, 4, 6, 8, 10 and 12 threads, in that order.
6. Checks the peak working set and appends the results to the CSV file.

Every configuration performs one warm-up execution followed by seven measured repetitions.

### Timing and statistics

- Time is measured with `omp_get_wtime()` around the multiplication only. The output matrix is cleared outside the timed interval.
- In the recorded data, all times are multiples of 1 ms, so the effective timer resolution is about 1 ms. This limits the precision of the 512 × 512 results.
- The median is the summary statistic for all comparisons, speedups and efficiencies.
- Minimum, maximum and the population standard deviation (divided by the number of repetitions) are also recorded.

### Thread control

The program disables dynamic team adjustment (`omp_set_dynamic(0)`), sets the thread count in code (`omp_set_num_threads`) and records the actual team size reported by `omp_get_num_threads()`. It stops if the team size changes between repetitions. No external numerical library is used, so there are no hidden library threads.

### Numeric type

All matrices use `double`. The numeric type is fixed, so it is documented here and is not a column of the CSV file.

## Correctness Validation

The program automatically runs these tests before benchmarking:

- A known 2 × 2 multiplication, checked against a hand-computed result for the independent IJK reference, the sequential code and the parallel code.
- Multiplication by a 3 × 3 identity matrix, for the sequential and parallel code.
- Multiplication by a 3 × 3 zero matrix, for the sequential and parallel code.

At the benchmark sizes (512, 1024 and 2048), the reference result is computed by the sequential IKJ function, and the output of each thread configuration is compared with it element by element. The independent IJK implementation is used only in the 2 × 2 test.

Values are accepted when `|a - b| <= 1e-12 + 1e-9 * |a|`, and non-finite values are rejected. The justification of these tolerances is in the report.

## Memory Budget

The controlled experiment uses a theoretical matrix-allocation budget of 256 MiB. Before allocating, the program estimates the memory required by the four matrix arrays (A, B, C and the reference):

```text
M = 4 * n^2 * sizeof(double)
```

Configurations that exceed the budget are rejected before allocation. After the benchmark, the program reads the process peak working set (`PeakWorkingSetSize` through `GetProcessMemoryInfo`) and checks it against the same budget. It does not save results if the budget is exceeded.

The budget is therefore applied through a pre-allocation check and a post-execution check. It is not a continuously enforced operating-system limit.

The peak working set is read once at the end of each run, so it has the same value for all thread counts of a size. It includes code, runtime data and other allocations, so it is not the exact memory used by the matrices.

## Runtime Limit

The controlled experiment declares a memory budget but no automatic runtime limit, and `parallel.c` implements no timeout. An exploratory attempt at 4096 × 4096 was stopped manually because it was taking too long. It produced no result and no execution time.

## Tested and Attempted Sizes

- 128 × 128: functional test only; execution times were too short for meaningful comparisons.
- 512 × 512, 1024 × 1024 and 2048 × 2048: completed benchmarks.
- 4096 × 4096: rejected by the pre-allocation check, because its four matrices need about 512 MiB, above the 256 MiB budget. No benchmark result exists for this size.

The largest size tested within the declared limits is 2048 × 2048. The largest size attempted is 4096 × 4096.

## Raw Data Format

The raw CSV file has one row per repetition and the columns:

```text
size, method, requested_threads, actual_threads, repetition, time_seconds,
median_seconds, min_seconds, max_seconds, stddev_seconds, speedup, efficiency,
peak_working_set_bytes
```

- `method` is `sequential_ikj` or `parallel_ikj`.
- The final dataset contains 168 rows: 3 matrix sizes × 8 configurations (the sequential baseline and 7 thread counts) × 7 repetitions.
- The summary file contains the best parallel configuration for each size.

The program appends measurements to the CSV file. Before running a new complete experiment, back up the existing dataset and prepare a clean output file. Do not append new measurements to the dataset used for the final analysis unless the additional runs are intentionally part of the experiment.

## Generating the Summary Table

```powershell
python .\generate_summary.py
```

The summary is saved to `results/assignment3/summary_results_limited.csv`. The script processes the raw measurements and calculates the median time, best parallel configuration, speedup and efficiency for each matrix size.

## Generating the Plots

```powershell
python .\generate_plots.py
```

The nine plots are saved in `plots/limited/`: execution time, speedup and parallel efficiency, each for 512 × 512, 1024 × 1024 and 2048 × 2048. In the execution-time plots, the error bars span the minimum to the maximum of the seven repetitions around the median. The plots are generated directly from the raw CSV file.

## Reproducibility Notes

- Sequential and parallel implementations use the same input matrices, numeric type, storage layout and algorithm. The differences between them are documented in the report.
- The fixed seed regenerates the same input matrices.
- The Python scripts regenerate the summary table and plots from the existing CSV file without running the benchmarks again:

```powershell
python .\generate_summary.py
python .\generate_plots.py
```

## Limitations

- All results come from one computer, and other processors, memory systems or schedulers may behave differently.
- CPU affinity was not configured, and CPU frequency, power mode and utilisation were not recorded.
- The timer resolution is about 1 ms, and each configuration has only seven repetitions, so no confidence intervals are reported.
- Each matrix size is one program run with a fixed configuration order, so run-order effects are not separated.
- The optional SIMD vectorisation extension was not carried out. GCC may auto-vectorise at `-O2`, and no evidence was collected either way, so the code should not be assumed to be non-vectorised. Both versions use the same compiler flags.

See `report/report.pdf` for the complete methodology, correctness validation, performance analysis, memory measurements, limitations and conclusions.