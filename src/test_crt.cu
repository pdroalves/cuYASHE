#include <NTL/ZZ_pEX.h>

#include "polynomial.h"
#include "ciphertext.h"
#include "yashe.h"
#include "settings.h"
#include "cuda_functions.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

NTL_CLIENT


#define BILLION  1000000000L
#define MILLION  1000000L
#define N 100

double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

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


int main(void){
    const int degree = 4096;
    Polynomial phi;
    ZZ_pX NTL_Phi;

    srand (36251);
    ZZ q;
    q = conv<ZZ>("1171313591017775093490277364417L");
    Polynomial::global_mod = q;
    ZZ_p::init(q); // Defines GF(q)

    Polynomial::BuildNthCyclotomic(&phi, degree); // generate an cyclotomic polynomial
    phi.set_mod(Polynomial::global_mod);
    Polynomial::global_phi = &phi;

    // Set params to NTL (just for comparison reasons)
    ZZ_p::init(Polynomial::global_mod);
    for(int i = 0; i <= phi.deg();i++){
      NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
    }
    ZZ_pE::init(NTL_Phi);

    Polynomial::gen_crt_primes(Polynomial::global_mod,degree);
    
    //////////
    struct timespec start, stop;

    Polynomial pA;

    Polynomial::random(&pA,degree-1);
    
    std::cout << "Starting..." << std::endl << std::endl;
    clock_gettime( CLOCK_REALTIME, &start);
    for(unsigned int i = 0; i < 100;i++){
      pA.crt();
      cudaError_t result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
      pA.set_crt_computed(false);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "Polynomial CRT: " << compute_time_ms(start,stop)/100 << std::endl;


    clock_gettime( CLOCK_REALTIME, &start);
    for(unsigned int i = 0; i < 100;i++){
      pA.update_device_data();
      cudaError_t result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
      pA.set_device_updated(false);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "Polynomial device update: " << compute_time_ms(start,stop)/100 << std::endl;

    pA.set_device_updated(true);
    pA.set_host_updated(false);

    clock_gettime( CLOCK_REALTIME, &start);
    for(unsigned int i = 0; i < 100;i++){
      pA.update_host_data();
      cudaError_t result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
      pA.set_host_updated(false);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "Polynomial host update: " << compute_time_ms(start,stop)/100 << std::endl;

    pA.set_host_updated(true);

    clock_gettime( CLOCK_REALTIME, &start);
    for(unsigned int i = 0; i < 100;i++){
      pA.icrt();
      cudaError_t result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
      pA.set_icrt_computed(false);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "Polynomial ICRT: " << compute_time_ms(start,stop)/100 << std::endl;

}
