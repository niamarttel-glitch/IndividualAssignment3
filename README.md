# Assignment 3 — Parallel Matrix Multiplication

## Project overview

This project evaluates the performance of dense matrix multiplication using sequential C code and OpenMP parallelisation.

The study compares execution time, speedup, parallel efficiency and memory consumption across different matrix sizes and thread counts.

The multiplication algorithm is IKJ in both implementations. Matrices use row-major storage and double-precision floating-point values.

## Project structure

```text
IA3/
├── src/
│   └── c/
│       ├── parallel.c
│       ├── parallel.exe
│       ├── test_openmp.c
│       └── test_openmp.exe
├── results/
│   └── assignment3/
│       ├── parallel_results.csv
│       └── summary_results.csv
├── plots/
│   ├── execution_time_512.png
│   ├── execution_time_1024.png
│   ├── execution_time_2048.png
│   ├── speedup_512.png
│   ├── speedup_1024.png
│   ├── speedup_2048.png
│   ├── efficiency_512.png
│   ├── efficiency_1024.png
│   └── efficiency_2048.png
├── report/
│   └── report.md
├── generate_plots.py
├── generate_summary.py
└── README.md
```

## Hardware and software

The experiments were performed on:

- CPU: 12th Gen Intel Core i5-12450H.
- Physical cores: 8.
- Logical processors: 12.
- RAM: 15.73 GB.
- Operating system: Windows.
- Compiler: GCC 16.2.0 (MSYS2 UCRT64).
- Parallelisation framework: OpenMP.
- Python: 3.14.2.
- Matplotlib: 3.10.8.

## Dependencies

Install MSYS2 with the UCRT64 GCC toolchain and OpenMP support.

Python dependencies can be installed with:

```powershell
python -m pip install matplotlib==3.10.8
```

The C program also uses the Windows Process Status API to measure peak working set.

## Compilation

Open PowerShell in the project root directory and configure the compiler path:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
```

Compile the parallel matrix multiplication program:

```powershell
gcc -O2 -std=c11 -Wall -Wextra -fopenmp .\src\c\parallel.c -o .\src\c\parallel.exe -lpsapi -lm
```

## Running the benchmarks

Run the program with the desired square matrix dimension:

```powershell
.\src\c\parallel.exe 512
.\src\c\parallel.exe 1024
.\src\c\parallel.exe 2048
```

The program generates deterministic input matrices using a fixed random seed. Each configuration performs one warm-up execution and seven measured repetitions.

The tested thread counts are 1, 2, 4, 6, 8, 10 and 12.

Benchmark measurements are written to:

```text
results/assignment3/parallel_results.csv
```

**Important:** the benchmark program writes measurements to the CSV file. To perform a fresh complete experiment, preserve a backup of existing results and start with a clean output file. Do not append a second set of repetitions to the existing dataset used by the analysis scripts.

## Generating the summary table

Run:

```powershell
python .\generate_summary.py
```

The summary is saved to:

```text
results/assignment3/summary_results.csv
```

## Generating the plots

Run:

```powershell
python .\generate_plots.py
```

The nine plots are saved in the `plots/` directory:

- Execution time.
- Speedup.
- Parallel efficiency.

Each metric is plotted separately for the three benchmark matrix sizes.

## Reproducibility notes

The program uses the same input matrices for sequential and parallel comparisons, the same numeric type and storage layout, and a fixed random seed.

The raw CSV contains the measurements used in the report. The Python scripts generate the summary table and plots from that dataset without rerunning the benchmarks.

See `report/report.md` for the experimental methodology, correctness validation, performance analysis, resource measurements and conclusions.