#include <stdio.h>
#include "common.h"
// #include "cuda_functions.h"
// #include "polynomial.h"
// #include "ciphertext.h"


// template Polynomial common_addition<Polynomial>(Polynomial *a,Polynomial *b);
// template Ciphertext common_addition<Ciphertext>(Ciphertext *a,Ciphertext *b);

uint64_t get_cycles() {
  unsigned int hi, lo;
  asm (
    "cpuid\n\t"/*serialize*/
    "rdtsc\n\t"/*read the clock*/
    "mov %%edx, %0\n\t"
    "mov %%eax, %1\n\t" 
    : "=r" (hi), "=r" (lo):: "%rax", "%rbx", "%rcx", "%rdx"
  );
  return ((uint64_t) lo) | (((uint64_t) hi) << 32);
}

#define BILLION  1000000000L
#define MILLION  1000000L
#define N 100

double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

const std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tstruct);

    return buf;
}