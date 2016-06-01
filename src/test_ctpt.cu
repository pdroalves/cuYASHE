/**
 * cuYASHE
 * Copyright (C) 2015-2016 cuYASHE Authors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
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
    uint64_t t;
    Yashe cipher;
    int degree;
    Polynomial phi;
    ZZ_pX NTL_Phi;

    srand (36251);

    // Params
    ZZ q;
    q = conv<ZZ>("1171313591017775093490277364417L");
    // q = conv<ZZ>("655615111");
    Polynomial::global_mod = q;
    ZZ_p::init(q); // Defines GF(q)

    t = 35951;
    degree = 4096;
    int w = 72;

    Polynomial::BuildNthCyclotomic(&phi, degree); // generate an cyclotomic polynomial
    phi.set_mod(Polynomial::global_mod);
    Polynomial::global_phi = &phi;

    // Set params to NTL (just for comparison reasons)
    ZZ_p::init(Polynomial::global_mod);
    for(int i = 0; i <= phi.deg();i++){
      NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
    }
    ZZ_pE::init(NTL_Phi);

    CUDAFunctions::init(2*degree);

    Polynomial::gen_crt_primes(Polynomial::global_mod,degree);

    // Yashe
    cipher = Yashe();

    Yashe::d = degree;
    Yashe::phi = phi;
    Yashe::q = q;
    // std::cout << ZZ_p::modulus() << std::endl;
    // std::cout << q << std::endl;

    Yashe::t = t;
    Yashe::w = w;
    Yashe::lwq = floor(NTL::log(q)/(log(2)*w)+1);

    cipher.generate_keys();

    //////////
    struct timespec start, stop;

    Ciphertext cA;
    Ciphertext cB;
    Polynomial pA;
    Polynomial pB;

    Ciphertext::random(&cA,degree-1);
    Ciphertext::random(&cB,degree-1);
    Polynomial::random(&pA,degree-1);
    Polynomial::random(&pB,degree-1);
    
    cA.update_device_data();
    cB.update_device_data();

    std::cout << "Starting..." << std::endl << std::endl;
    clock_gettime( CLOCK_REALTIME, &start);
    for(unsigned int i = 0; i < 100;i++){
      Ciphertext c = cA+cB;
      cudaError_t result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "Ciphertext sum: " << compute_time_ms(start,stop)/100 << std::endl;


    clock_gettime( CLOCK_REALTIME, &start);
    for(unsigned int i = 0; i < 100;i++){
      cA += cB;
      cudaError_t result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "Ciphertext inplace sum: " << compute_time_ms(start,stop)/100 << std::endl;


    pA.update_device_data();
    pB.update_device_data();

    clock_gettime( CLOCK_REALTIME, &start);
    for(unsigned int i = 0; i < 100;i++){
      Polynomial p = pA+pB;
      cudaError_t result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "Polynomial sum: " << compute_time_ms(start,stop)/100 << std::endl;


    clock_gettime( CLOCK_REALTIME, &start);
    for(unsigned int i = 0; i < 100;i++){
      pA += pB;
      cudaError_t result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "Polynomial inplace sum: " << compute_time_ms(start,stop)/100 << std::endl;

}
