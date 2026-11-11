
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <windows.h>
#include <psapi.h>
#include <limits.h>
#include <errno.h>

#define REPETITIONS 7
#define SEED 123456789ULL
#define ABS_TOLERANCE 1e-12
#define REL_TOLERANCE 1e-9
#define DEFAULT_SIZE 1024
#define MAX_CONFIGS 8

typedef struct {
    double median;
    double minimum;
    double maximum;
    double stddev;
} Statistics;

/* Reproducible pseudo-random number generator. */
static uint64_t next_random(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Allocate a square matrix in row-major order. */
static double *allocate_matrix(size_t n) {
    if (n == 0 || n > SIZE_MAX / n ||
        n * n > SIZE_MAX / sizeof(double)) {
        return NULL;
    }

    return (double *)calloc(n * n, sizeof(double));
}

/* Generate matrix values. */
static void generate_matrix(double *matrix, size_t n,
                            uint64_t *state) {
    for (size_t i = 0; i < n * n; i++) {
        matrix[i] =
            (double)(next_random(state) % 1000) / 1000.0;
    }
}

/* Reset output matrix before each multiplication. */
static void clear_matrix(double *matrix, size_t n) {
    memset(matrix, 0, n * n * sizeof(double));
}

/* Sequential IKJ matrix multiplication. */
static void multiply_sequential(const double *A,
                                const double *B,
                                double *C,
                                size_t n) {
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            double value = A[i * n + k];

            for (size_t j = 0; j < n; j++) {
                C[i * n + j] += value * B[k * n + j];
            }
        }
    }
}

/* Parallel IKJ matrix multiplication using OpenMP. */
static void multiply_parallel(const double *A,
                              const double *B,
                              double *C,
                              size_t n,
                              int *actual_threads) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            *actual_threads = omp_get_num_threads();
        }

        #pragma omp for schedule(static)
        for (long long i = 0; i < (long long)n; i++) {
            for (size_t k = 0; k < n; k++) {
                double value = A[(size_t)i * n + k];

                for (size_t j = 0; j < n; j++) {
                    C[(size_t)i * n + j] +=
                        value * B[k * n + j];
                }
            }
        }
    }
}

/* Compare two matrices using absolute and relative tolerances. */
static int matrices_equal(const double *A,
                          const double *B,
                          size_t n) {
    for (size_t i = 0; i < n * n; i++) {
        /* Reject NaN and infinite values. */
        if (!isfinite(A[i]) || !isfinite(B[i])) {
            return 0;
        }

        double difference = fabs(A[i] - B[i]);

        double limit = ABS_TOLERANCE
                     + REL_TOLERANCE * fabs(A[i]);

        if (difference > limit) {
            return 0;
        }
    }

    return 1;
}

/* Test sequential and parallel multiplication with known results. */
static int run_basic_tests(void) {
    const size_t n = 3;

    double identity[9] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    double zero[9] = {0};

    double B[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    double expected_zero[9] = {0};
    double sequential[9] = {0};
    double parallel[9] = {0};

    int actual_threads = 0;

    omp_set_dynamic(0);
    omp_set_num_threads(2);

    /* Test 1: Identity multiplied by B must equal B. */
    multiply_sequential(identity, B, sequential, n);
    multiply_parallel(identity, B, parallel, n, &actual_threads);

    if (!matrices_equal(B, sequential, n) ||
        !matrices_equal(B, parallel, n)) {
        printf("Basic test FAILED: identity matrix.\n");
        return 0;
    }

    /* Test 2: Zero multiplied by B must produce zero. */
    clear_matrix(sequential, n);
    clear_matrix(parallel, n);

    multiply_sequential(zero, B, sequential, n);
    multiply_parallel(zero, B, parallel, n, &actual_threads);

    if (!matrices_equal(expected_zero, sequential, n) ||
        !matrices_equal(expected_zero, parallel, n)) {
        printf("Basic test FAILED: zero matrix.\n");
        return 0;
    }

    printf("Basic tests: identity and zero matrices PASSED.\n");
    return 1;
}

/* Sort execution times. */
static void sort_times(double *times, int count) {
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (times[j] < times[i]) {
                double temp = times[i];
                times[i] = times[j];
                times[j] = temp;
            }
        }
    }
}

/* Calculate median, minimum, maximum and population standard deviation. */
static Statistics calculate_statistics(const double *times) {
    Statistics stats;
    double sorted[REPETITIONS];
    double sum = 0.0;
    double variance = 0.0;

    for (int i = 0; i < REPETITIONS; i++) {
        sorted[i] = times[i];
        sum += times[i];
    }

    sort_times(sorted, REPETITIONS);

    stats.minimum = sorted[0];
    stats.median = sorted[REPETITIONS / 2];
    stats.maximum = sorted[REPETITIONS - 1];

    double mean = sum / REPETITIONS;

    for (int i = 0; i < REPETITIONS; i++) {
        double difference = times[i] - mean;
        variance += difference * difference;
    }

    stats.stddev = sqrt(variance / REPETITIONS);

    return stats;
}

/* Benchmark sequential implementation. */
static void benchmark_sequential(const double *A,
                                 const double *B,
                                 double *C,
                                 size_t n,
                                 double *times) {
    /* Warm-up execution. */
    clear_matrix(C, n);
    multiply_sequential(A, B, C, n);

    /* Measured executions. */
    for (int r = 0; r < REPETITIONS; r++) {
        clear_matrix(C, n);

        double start = omp_get_wtime();
        multiply_sequential(A, B, C, n);
        double end = omp_get_wtime();

        times[r] = end - start;
    }
}

/* Benchmark parallel implementation and verify thread consistency. */
static int benchmark_parallel(const double *A,
                              const double *B,
                              double *C,
                              size_t n,
                              int requested_threads,
                              double *times) {
    int actual_threads = 0;

    omp_set_num_threads(requested_threads);

    /* Warm-up execution. */
    clear_matrix(C, n);
    multiply_parallel(A, B, C, n, &actual_threads);

    int expected_threads = actual_threads;

    if (expected_threads <= 0) {
        fprintf(stderr, "Could not determine actual thread count.\n");
        return 0;
    }

    /* Measured executions. */
    for (int r = 0; r < REPETITIONS; r++) {
        clear_matrix(C, n);

        int run_threads = 0;

        double start = omp_get_wtime();
        multiply_parallel(A, B, C, n, &run_threads);
        double end = omp_get_wtime();

        /* Ensure the number of threads remains constant. */
        if (run_threads != expected_threads) {
            fprintf(stderr,
                    "Thread count changed: expected %d, got %d.\n",
                    expected_threads, run_threads);
            return 0;
        }

        times[r] = end - start;
    }

    return expected_threads;
}

/* Retrieve peak process working set on Windows. */
static size_t get_peak_memory_bytes(void) {
    PROCESS_MEMORY_COUNTERS counters;

    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             &counters,
                             sizeof(counters))) {
        return (size_t)counters.PeakWorkingSetSize;
    }

    return 0;
}

/* Save experimental results without overwriting previous runs. */
static int save_results(
    size_t n,
    const int *thread_configs,
    const int *actual_configs,
    const double times[MAX_CONFIGS][REPETITIONS],
    const Statistics *stats,
    double sequential_median,
    size_t peak_memory) {

    FILE *file = fopen(
        "results/assignment3/parallel_results.csv", "a+");

    if (file == NULL) {
        perror("Could not open CSV results file");
        return 0;
    }

    /* Write CSV header only if the file is empty. */
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 0;
    }

    if (ftell(file) == 0) {
        fprintf(file,
            "size,method,requested_threads,actual_threads,"
            "repetition,time_seconds,median_seconds,"
            "min_seconds,max_seconds,stddev_seconds,"
            "speedup,efficiency,peak_working_set_bytes\n");
    }

    for (int config = 0; config < MAX_CONFIGS; config++) {
        double speedup =
            sequential_median / stats[config].median;

        double efficiency = 0.0;

        if (actual_configs[config] > 0) {
            efficiency =
                speedup / (double)actual_configs[config];
        }

        for (int r = 0; r < REPETITIONS; r++) {
            const char *method =
                (config == 0)
                ? "sequential_ikj"
                : "parallel_ikj";

            fprintf(file,
                "%zu,%s,%d,%d,%d,%.9f,%.9f,"
                "%.9f,%.9f,%.9f,%.6f,%.6f,%zu\n",
                n,
                method,
                thread_configs[config],
                actual_configs[config],
                r + 1,
                times[config][r],
                stats[config].median,
                stats[config].minimum,
                stats[config].maximum,
                stats[config].stddev,
                speedup,
                efficiency,
                peak_memory);
        }
    }

    int success = !ferror(file);
    fclose(file);

    return success;
}

int main(int argc, char *argv[]) {
    /* Run basic correctness tests before benchmarking. */
    if (!run_basic_tests()) {
        return EXIT_FAILURE;
    }

    size_t n = DEFAULT_SIZE;

    /* Read matrix size from command line. */
    if (argc >= 2) {
        char *end;
        errno = 0;

        unsigned long long parsed = strtoull(argv[1], &end, 10);

        if (*argv[1] == '\0' ||
            *argv[1] == '-' ||
            *end != '\0' ||
            errno == ERANGE ||
            parsed == 0 ||
            parsed > SIZE_MAX ||
            parsed > LLONG_MAX) {

            fprintf(stderr, "Invalid matrix size.\n");
            return EXIT_FAILURE;
        }

        n = (size_t)parsed;
    }

    /*
       First entry is the sequential baseline.
       Second entry is parallel execution with one thread.
    */
    const int thread_configs[MAX_CONFIGS] =
        {1, 1, 2, 4, 6, 8, 10, 12};

    int actual_configs[MAX_CONFIGS] = {0};

    double times[MAX_CONFIGS][REPETITIONS] = {{0}};

    Statistics stats[MAX_CONFIGS];

    omp_set_dynamic(0);

    printf("\n========================================\n");
    printf(" ASSIGNMENT 3 - PARALLEL MATRIX MULTIPLICATION\n");
    printf("========================================\n");

    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Numeric type: double\n");
    printf("Algorithm: IKJ\n");
    printf("Storage: row-major\n");
    printf("OpenMP dynamic teams: disabled\n");
    printf("Warm-up executions: 1\n");
    printf("Measured repetitions: %d\n", REPETITIONS);

    /* Allocate matrices. */
    double *A = allocate_matrix(n);
    double *B = allocate_matrix(n);
    double *C = allocate_matrix(n);
    double *reference = allocate_matrix(n);

    if (A == NULL || B == NULL || C == NULL ||
        reference == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");

        free(A);
        free(B);
        free(C);
        free(reference);

        return EXIT_FAILURE;
    }

    /* Generate reproducible input matrices. */
    uint64_t random_state = SEED;

    generate_matrix(A, n, &random_state);
    generate_matrix(B, n, &random_state);

    /* Calculate sequential reference result. */
    printf("\nGenerating sequential reference...\n");

    multiply_sequential(A, B, reference, n);

    /* Sequential benchmark. */
    printf("\n--- Sequential benchmark ---\n");

    benchmark_sequential(A, B, C, n, times[0]);

    actual_configs[0] = 1;
    stats[0] = calculate_statistics(times[0]);

    if (!matrices_equal(reference, C, n)) {
        fprintf(stderr, "Sequential validation FAILED.\n");

        free(A);
        free(B);
        free(C);
        free(reference);

        return EXIT_FAILURE;
    }

    printf("Sequential median: %.6f seconds\n",
           stats[0].median);

    printf("Sequential validation: OK\n");

    /* Parallel benchmarks. */
    printf("\n--- Parallel benchmarks ---\n");

    for (int config = 1; config < MAX_CONFIGS; config++) {
        int requested = thread_configs[config];

        actual_configs[config] =
            benchmark_parallel(A, B, C, n, requested,
                               times[config]);

        /* Stop if the benchmark or thread consistency check failed. */
        if (actual_configs[config] <= 0) {
            fprintf(stderr,
                    "Parallel benchmark failed for %d requested threads.\n",
                    requested);

            free(A);
            free(B);
            free(C);
            free(reference);

            return EXIT_FAILURE;
        }

        stats[config] =
            calculate_statistics(times[config]);

        /* Validate the final parallel result. */
        if (!matrices_equal(reference, C, n)) {
            fprintf(stderr,
                    "Validation FAILED with %d threads.\n",
                    requested);

            free(A);
            free(B);
            free(C);
            free(reference);

            return EXIT_FAILURE;
        }

        double speedup =
            stats[0].median / stats[config].median;

        double efficiency =
            speedup / (double)actual_configs[config];

        printf("\nRequested threads: %d\n", requested);
        printf("Actual threads: %d\n",
               actual_configs[config]);

        printf("Median time: %.6f seconds\n",
               stats[config].median);

        printf("Standard deviation: %.6f seconds\n",
               stats[config].stddev);

        printf("Speedup: %.3fx\n", speedup);

        printf("Efficiency: %.2f%%\n",
               efficiency * 100.0);

        printf("Validation: OK\n");
    }

    /* Retrieve peak memory. */
    size_t peak_memory = get_peak_memory_bytes();

    printf("\n========================================\n");
    printf(" EXPERIMENT SUMMARY\n");
    printf("========================================\n");

    printf("Sequential median: %.6f seconds\n",
           stats[0].median);

    printf("Peak working set: %.2f MiB\n",
           (double)peak_memory / (1024.0 * 1024.0));

    /* Append experimental results to CSV. */
    if (!save_results(n,
                      thread_configs,
                      actual_configs,
                      times,
                      stats,
                      stats[0].median,
                      peak_memory)) {
        fprintf(stderr, "Could not save CSV results.\n");

        free(A);
        free(B);
        free(C);
        free(reference);

        return EXIT_FAILURE;
    }

    printf("Results saved to results/assignment3/parallel_results.csv\n");

    /* Release allocated memory. */
    free(A);
    free(B);
    free(C);
    free(reference);

    return EXIT_SUCCESS;
}
