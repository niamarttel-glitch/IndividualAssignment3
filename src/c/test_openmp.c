
#include <stdio.h>
#include <omp.h>

int main(void) {

    omp_set_dynamic(0);

    #pragma omp parallel
    {
        printf("Hello from thread %d of %d\n",
               omp_get_thread_num(),
               omp_get_num_threads());
    }

    return 0;
}

