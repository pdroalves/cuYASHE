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