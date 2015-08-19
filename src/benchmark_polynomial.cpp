#include "benchmark_polynomial.h"
#include "polynomial.h"
#include "common.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <time.h>
#include <unistd.h>
#include <NTL/ZZ_pEX.h>
#include <iomanip>
 #include <stdint.h>

#define BILLION  1000000000L
#define MILLION  1000000L
#define N 100

double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

int main(void){
  cout << fixed;
  cout.precision(2);

  ZZ q;
  NTL::power2(q,127);
  q -= conv<ZZ>("1");

  struct timespec start, stop;
  clock_gettime( CLOCK_REALTIME, &start);
  sleep(1);
  clock_gettime( CLOCK_REALTIME, &stop);
  std::cout << "1 sec: " << compute_time_ms(start,stop) << std::endl;

  Polynomial::global_mod = q; // Defines default GF(q)
  ZZ_p::init(q); // Defines GF(q)
  Polynomial phi;
  phi.set_mod(Polynomial::global_mod);

  // for(int d = 32;d <= 4096;d *= 2){
  for(int d = 256;d <= 256;d *= 2){
    std::cout << "d: " << d << std::endl;

    clock_gettime( CLOCK_REALTIME, &start);

    Polynomial::BuildNthCyclotomic(&phi, d); // generate an cyclotomic polynomial

    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "Irreducible polynomial generated in " << compute_time_ms(start,stop) << " ms." << std::endl;
    std::cout << "Generating " << phi.deg() << " degree polynomials." << std::endl;

    Polynomial::global_phi = &phi;

    clock_gettime( CLOCK_REALTIME, &start);
    Polynomial::gen_crt_primes(Polynomial::global_mod,d);
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "CRT primes generated in " << compute_time_ms(start,stop) << " ms." << std::endl;

    std::cout << "q: " << NTL::NumBytes(q)*8 << " bits" << std::endl;

    ///////////////////////////////////////////////
    // ADD
    // Time measured with memory copy
    Polynomial a;
    Polynomial b;
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){

      Polynomial c = (a+b);
      a.set_device_updated(false);
      b.set_device_updated(false);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "ADD) Time measured with memory copy: " << compute_time_ms(start,stop)/N << " ms" << std::endl;

    // Time measured without memory copy
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);

    a.crt();
    a.update_device_data();
    b.crt();
    b.update_device_data();

    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      Polynomial c = (a+b);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "ADD) Time measured without memory copy: " << compute_time_ms(start,stop)/N << " ms" << std::endl;

    ///////////////////////////////////////////////
    // MUL
    // Time measured with memory copy
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){

      Polynomial c = (a*b);
      a.set_device_updated(false);
      b.set_device_updated(false);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "MUL) Time measured with memory copy: " << compute_time_ms(start,stop)/N << " ms" << std::endl;

    // Time measured without memory copy
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);

    a.crt();
    a.update_device_data();
    b.crt();
    b.update_device_data();

    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      Polynomial c = (a*b);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << "MUL) Time measured without memory copy: " << compute_time_ms(start,stop)/N << " ms" << std::endl;
  }
}
