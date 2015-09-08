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



// Get current date/time, format is YYYYMMDDHHmmss
const std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y%m%d%H%M%S", &tstruct);

    return buf;
}

int main(int argc,char* argv[]){
  cout << fixed;
  cout.precision(2);

  ofstream copyHtD;
  ofstream copyDtH;
  ofstream crt;
  ofstream icrt;
  ofstream gpu_add_without_memcopy;
  ofstream gpu_mult_without_memcopy;
  ofstream gpu_add_with_memcopy;
  ofstream gpu_mult_with_memcopy;
  std::string copyHtD_filename;
  std::string copyDtH_filename;
  std::string crt_filename;
  std::string icrt_filename;
  std::string gpu_add_without_memcopy_filename;
  std::string gpu_mult_without_memcopy_filename;
  std::string gpu_add_with_memcopy_filename;
  std::string gpu_mult_with_memcopy_filename;

  if(argc == 2){
    char* suffix = argv[1];

    copyHtD_filename = "copyHtD_"+current_date_time()+"_"+suffix+".dat";
    copyDtH_filename = "copyDtH_"+current_date_time()+"_"+suffix+".dat";
    crt_filename = "crt_"+current_date_time()+"_"+suffix+".dat";
    icrt_filename = "icrt_"+current_date_time()+"_"+suffix+".dat";
    gpu_add_without_memcopy_filename = "gpu_add_with_memcopy_without_memcopy_"+current_date_time()+"_"+suffix+".dat";
    gpu_mult_without_memcopy_filename = "gpu_mult_without_memcopy_"+current_date_time()+"_"+suffix+".dat";
    gpu_add_with_memcopy_filename = "gpu_add_with_memcopy_"+current_date_time()+"_"+suffix+".dat";
    gpu_mult_with_memcopy_filename = "gpu_mult_with_memcopy_"+current_date_time()+"_"+suffix+".dat";
  }else{   

    copyHtD_filename = "copyHtD_"+current_date_time()+".dat";
    copyDtH_filename = "copyDtH_"+current_date_time()+".dat";
    crt_filename = "crt_"+current_date_time()+".dat";
    icrt_filename = "icrt_"+current_date_time()+".dat";
    gpu_add_without_memcopy_filename = "gpu_add_with_memcopy_without_memcopy_"+current_date_time()+".dat";
    gpu_mult_without_memcopy_filename = "gpu_mult_without_memcopy_"+current_date_time()+".dat";
    gpu_add_with_memcopy_filename = "gpu_add_with_memcopy_"+current_date_time()+".dat";
    gpu_mult_with_memcopy_filename = "gpu_mult_with_memcopy_"+current_date_time()+".dat";
  }
  copyHtD.open (copyHtD_filename);
  copyDtH.open (copyDtH_filename);
  crt.open (crt_filename);
  icrt.open (icrt_filename);
  gpu_add_without_memcopy.open (gpu_add_without_memcopy_filename);
  gpu_mult_without_memcopy.open (gpu_mult_without_memcopy_filename);
  gpu_add_with_memcopy.open (gpu_add_with_memcopy_filename);
  gpu_mult_with_memcopy.open (gpu_mult_with_memcopy_filename);

  std::cout << "Writing copyHtD data to " << copyHtD_filename << std::endl;
  std::cout << "Writing copyDtH data to " << copyDtH_filename << std::endl;
  std::cout << "Writing crt data to " << crt_filename << std::endl;
  std::cout << "Writing icrt data to " << icrt_filename << std::endl;
  std::cout << "Writing gpu_add_without_memcopy data to " << gpu_add_without_memcopy_filename << std::endl;
  std::cout << "Writing gpu_mult_without_memcopy data to " << gpu_mult_without_memcopy_filename << std::endl;
  std::cout << "Writing gpu_add_with_memcopy data to " << gpu_add_with_memcopy_filename << std::endl;
  std::cout << "Writing gpu_mult_with_memcopy data to " << gpu_mult_with_memcopy_filename << std::endl;
     
  ZZ q;
  NTL::power2(q,127);
  q -= conv<ZZ>("1");

  struct timespec start, stop;
  float diff;
  clock_gettime( CLOCK_REALTIME, &start);
  sleep(1);
  clock_gettime( CLOCK_REALTIME, &stop);
  diff = compute_time_ms(start,stop)/N;
  std::cout << "1 sec: " << diff << std::endl;

  Polynomial::global_mod = q; // Defines default GF(q)
  ZZ_p::init(q); // Defines GF(q)
  Polynomial phi;
  phi.set_mod(Polynomial::global_mod);

  for(int d = 512;d <= 4096;d *= 2){
    CUDAFunctions::init(2*d);

    std::cout << "d: " << d << std::endl;

    clock_gettime( CLOCK_REALTIME, &start);
    Polynomial::BuildNthCyclotomic(&phi, d); // generate an cyclotomic polynomial
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop);
    std::cout << "Irreducible polynomial generated in " << diff << " ms." << std::endl;
    std::cout << "Generating " << phi.deg() << " degree polynomials." << std::endl;

    Polynomial::global_phi = &phi;

    clock_gettime( CLOCK_REALTIME, &start);
    Polynomial::gen_crt_primes(Polynomial::global_mod,d);
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop);
    std::cout << "CRT primes generated in " << diff << " ms." << std::endl;

    std::cout << "q: " << NTL::NumBytes(q)*8 << " bits" << std::endl;

    Polynomial a;
    Polynomial b;
    ///////////////////////////////////////////////
    // Copy
    //
    Polynomial::random(&a,d-1);
    a.set_device_updated(false);
    a.crt();
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      a.update_device_data();
      a.set_device_updated(false);
      cudaDeviceSynchronize();
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "Copy) Host to Device: " << diff << " ms" << std::endl;

    Polynomial::random(&a,d-1);
    a.crt();
    a.update_device_data();
    a.set_host_updated(false);
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      a.update_host_data();
      a.set_host_updated(false);
      cudaDeviceSynchronize();
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "Copy) Device to host: " << diff << " ms" << std::endl;

    ///////////////////////////////////////////////
    // CRT/ICRT
    //
    Polynomial::random(&a,d-1);
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      a.set_device_updated(false);
      a.crt();
      cudaDeviceSynchronize();
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "CRT) Foward: " << diff << " ms" << std::endl;
    crt << N << diff  << std::endl;

    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      a.set_host_updated(false);
      a.icrt();
      cudaDeviceSynchronize();
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "CRT) Inverse: " << diff << " ms" << std::endl;
    icrt << N << diff  << std::endl;

    ///////////////////////////////////////////////
    // ADD
    // Time measured with memory copy
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){

      Polynomial c = (a+b);
      a.set_device_updated(false);
      b.set_device_updated(false);
      cudaDeviceSynchronize();

    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "ADD) Time measured with memory copy: " << diff << " ms" << std::endl;
    gpu_add_with_memcopy << N << diff  << std::endl;
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
      cudaDeviceSynchronize();

    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "ADD) Time measured without memory copy: " << diff << " ms" << std::endl;
    gpu_add_without_memcopy << N << diff  << std::endl;

    ///////////////////////////////////////////////
    // MUL
    // Time measured with memory copy
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);
    a.update_crt_spacing(2*d);
    b.update_crt_spacing(2*d);
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){

      Polynomial c = (a*b);
      a.set_device_updated(false);
      b.set_device_updated(false);
      cudaDeviceSynchronize();

    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "MUL) Time measured with memory copy: " << diff << " ms" << std::endl;
    gpu_mult_with_memcopy << N << diff  << std::endl;

    // Time measured without memory copy
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);
    a.update_crt_spacing(2*d);
    b.update_crt_spacing(2*d);
    a.crt();
    a.update_device_data();
    b.crt();
    b.update_device_data();

    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      Polynomial c = (a*b);
      cudaDeviceSynchronize();

    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "MUL) Time measured without memory copy: " << diff << " ms" << std::endl;
    gpu_mult_without_memcopy << N << diff  << std::endl;

  }
}
