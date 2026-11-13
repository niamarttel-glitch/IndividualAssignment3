# Assignment 3 — Parallel Matrix Multiplication

## Project Overview

This project evaluates the performance of dense matrix multiplication using a sequential C implementation and an OpenMP parallel implementation.

The objective is to investigate how the number of threads affects execution time, speedup, parallel efficiency and memory consumption for different matrix sizes.

Both implementations use the same IKJ multiplication algorithm, row-major storage and double-precision floating-point values (`double`). The project also includes correctness tests and an independent reference implementation using the IJK loop ordering.

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
│   └── report.md
├── generate_plots.py
├── generate_summary.py
└── README.md
```

Compiled executables are not included in the repository. They can be generated using the compilation instructions below.

## Hardware and Software

The experiments were performed on the following system:

| Component | Specification |
|---|---|
| CPU | 12th Gen Intel Core i5-12450H |
| Physical cores | 8 |
| Logical processors | 12 |
| RAM | 15.73 GB |
| Operating system | Windows |
| Compiler | GCC 16.2.0 (MSYS2 UCRT64) |
| Parallelisation framework | OpenMP |
| Python | 3.14.2 |
| Matplotlib | 3.10.8 |

No manual processor affinity was configured. Thread scheduling was handled by the operating system.

## Dependencies

The project requires:

- Windows operating system.
- MSYS2 with the UCRT64 GCC toolchain.
- GCC with OpenMP support.
- Python 3.14.2.
- Matplotlib 3.10.8.

Install Matplotlib using:

```powershell
python -m pip install matplotlib==3.10.8
```

The C program uses the Windows Process Status API to measure the process peak working set. This requires linking against the Windows `psapi` library.

## Compilation

Open PowerShell in the project root directory and configure the compiler path:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
```

Compile the parallel matrix multiplication program:

```powershell
gcc -O2 -std=c11 -Wall -Wextra -fopenmp .\src\c\parallel.c -o .\src\c\parallel.exe -lpsapi -lm
```

## Running the Benchmarks

Run the program with the desired square matrix dimension:

```powershell
.\src\c\parallel.exe 512
.\src\c\parallel.exe 1024
.\src\c\parallel.exe 2048
```

The program generates deterministic input matrices using a fixed random seed (`123456789`).

Each configuration performs one warm-up execution followed by seven measured repetitions. The median execution time is used for the main performance comparisons. Minimum, maximum and standard deviation are also recorded.

The tested parallel thread counts are:

- 1
- 2
- 4
- 6
- 8
- 10
- 12

The program automatically performs correctness tests, including:

- A known 2 × 2 matrix multiplication.
- Multiplication by an identity matrix.
- Multiplication by a zero matrix.
- Comparison against an independent IJK reference implementation.

### Memory Budget

The controlled experiment uses a theoretical matrix-allocation budget of 256 MiB. Before allocating the matrices, the program estimates the memory required for the four matrix arrays and rejects configurations that exceed this budget.

After the benchmark, the program measures the process peak working set using the Windows Process Status API and checks it against the same budget.

The memory budget is therefore applied through a pre-allocation check and a post-execution verification; it is not a continuously enforced operating-system memory limit.

Benchmark measurements are written to:

```text
results/assignment3/parallel_results_limited.csv
```

**Important:** The benchmark program appends measurements to the CSV file. Before running a new complete experiment, back up the existing dataset and prepare a clean output file. Do not append new measurements to the dataset used for the final analysis unless the additional runs are intentionally part of the experiment.

## Generating the Summary Table

Run:

```powershell
python .\generate_summary.py
```

The summary is saved to:

```text
results/assignment3/summary_results_limited.csv
```

The script processes the raw benchmark measurements and calculates the median execution time, best parallel configuration, speedup and parallel efficiency for each matrix size.

## Generating the Plots

Run:

```powershell
python .\generate_plots.py
```

The nine plots are saved in:

```text
plots/limited/
```

The generated plots cover:

- Execution time.
- Speedup.
- Parallel efficiency.

Each metric is plotted separately for the three benchmark matrix sizes: 512 × 512, 1024 × 1024 and 2048 × 2048.

The plots are generated directly from the raw CSV dataset.

## Experimental Results

The controlled benchmark campaign covers three matrix dimensions:

- 512 × 512.
- 1024 × 1024.
- 2048 × 2048.

The 2048 × 2048 matrix is the largest successfully completed benchmark size.

A 4096 × 4096 matrix requires approximately 512 MiB for the four matrix arrays used by the program. Since this exceeds the declared 256 MiB matrix-allocation budget, the controlled experiment rejects this configuration before allocation.

The final dataset contains 168 measurements: eight configurations (one sequential baseline and seven parallel thread counts), three matrix sizes and seven repetitions per configuration.

### Best Parallel Configurations

| Matrix size | Best thread count | Best execution time (s) | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| 512 × 512 | 6 | 0.011 | 4.091× | 68.18% |
| 1024 × 1024 | 8 | 0.085 | 5.859× | 73.24% |
| 2048 × 2048 | 12 | 1.001 | 4.074× | 33.95% |

The results show that parallel execution improves performance for all three tested matrix sizes. However, the best thread count depends on the matrix dimension. Increasing the number of threads does not always improve performance, since parallel overhead and resource contention can reduce the benefits of additional threads.

The efficiency also decreases for the largest matrix, indicating that the speedup does not scale proportionally with the number of threads.

See `report/report.md` for the complete experimental methodology, correctness validation, performance analysis, memory measurements, limitations and conclusions.

## Reproducibility Notes

The sequential and parallel implementations use the same input matrices, numeric type, storage layout and multiplication algorithm.

The fixed random seed ensures reproducible input data. The benchmark records the measurements used in the final analysis in the raw CSV file.

The Python scripts generate the summary table and plots from the existing dataset. They do not require the C benchmarks to be executed again.

To reproduce the analysis without repeating the computationally expensive benchmarks, run:

```powershell
python .\generate_summary.py
python .\generate_plots.py
```
