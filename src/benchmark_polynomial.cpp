#include "benchmark_polynomial.h"
#include "polynomial.h"
#include "settings.h"
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
#include "common.h"

#define BILLION  1000000000L
#define MILLION  1000000L
#define N 100


int main(int argc,char* argv[]){
  std::cout << "Starting. N = " << N << std::endl;

  size_t f, t;
  cudaSetDevice(0);

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
  ofstream gpu_reduce;
  ofstream cpu_reduce;
  ofstream modq;
  std::string copyHtD_filename;
  std::string copyDtH_filename;
  std::string crt_filename;
  std::string icrt_filename;
  std::string gpu_add_without_memcopy_filename;
  std::string gpu_mult_without_memcopy_filename;
  std::string gpu_add_with_memcopy_filename;
  std::string gpu_mult_with_memcopy_filename;
  std::string gpu_reduce_filename;
  std::string cpu_reduce_filename;
  std::string modq_filename;

  cudaError_t result;

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
    gpu_reduce_filename = "gpu_reduce_"+current_date_time()+"_"+suffix+".dat";
    cpu_reduce_filename = "cpu_reduce_"+current_date_time()+"_"+suffix+".dat";
    modq_filename = "modq_"+current_date_time()+"_"+suffix+".dat";
  }else{

    copyHtD_filename = "copyHtD_"+current_date_time()+".dat";
    copyDtH_filename = "copyDtH_"+current_date_time()+".dat";
    crt_filename = "crt_"+current_date_time()+".dat";
    icrt_filename = "icrt_"+current_date_time()+".dat";
    gpu_add_without_memcopy_filename = "gpu_add_with_memcopy_without_memcopy_"+current_date_time()+".dat";
    gpu_mult_without_memcopy_filename = "gpu_mult_without_memcopy_"+current_date_time()+".dat";
    gpu_add_with_memcopy_filename = "gpu_add_with_memcopy_"+current_date_time()+".dat";
    gpu_mult_with_memcopy_filename = "gpu_mult_with_memcopy_"+current_date_time()+".dat";
    gpu_reduce_filename = "gpu_reduce_"+current_date_time()+".dat";
    cpu_reduce_filename = "cpu_reduce_"+current_date_time()+".dat";
    modq_filename = "modq_"+current_date_time()+".dat";
  }
  copyHtD.open (copyHtD_filename);
  copyDtH.open (copyDtH_filename);
  crt.open (crt_filename);
  icrt.open (icrt_filename);
  gpu_add_without_memcopy.open (gpu_add_without_memcopy_filename);
  gpu_mult_without_memcopy.open (gpu_mult_without_memcopy_filename);
  gpu_add_with_memcopy.open (gpu_add_with_memcopy_filename);
  gpu_mult_with_memcopy.open (gpu_mult_with_memcopy_filename);
  gpu_reduce.open (gpu_reduce_filename);
  cpu_reduce.open (cpu_reduce_filename);
  modq.open (modq_filename);

  std::cout << "Writing copyHtD data to " << copyHtD_filename << std::endl;
  std::cout << "Writing copyDtH data to " << copyDtH_filename << std::endl;
  std::cout << "Writing crt data to " << crt_filename << std::endl;
  std::cout << "Writing icrt data to " << icrt_filename << std::endl;
  std::cout << "Writing gpu_add_without_memcopy data to " << gpu_add_without_memcopy_filename << std::endl;
  std::cout << "Writing gpu_mult_without_memcopy data to " << gpu_mult_without_memcopy_filename << std::endl;
  std::cout << "Writing gpu_add_with_memcopy data to " << gpu_add_with_memcopy_filename << std::endl;
  std::cout << "Writing gpu_mult_with_memcopy data to " << gpu_mult_with_memcopy_filename << std::endl;
  std::cout << "Writing gpu_reduce data to " << gpu_reduce_filename << std::endl;
  std::cout << "Writing cpu_reduce data to " << cpu_reduce_filename << std::endl;
  std::cout << "Writing modq data to " << modq_filename << std::endl;

  ZZ_pX NTL_Phi;
  ZZ q;
  NTL::power2(q,127);
  q -= conv<ZZ>("1");

  struct timespec start, stop;
  float diff;
  clock_gettime( CLOCK_REALTIME, &start);
  sleep(1);
  clock_gettime( CLOCK_REALTIME, &stop);
  diff = compute_time_ms(start,stop);
  std::cout << "1 sec: " << diff << std::endl;

  Polynomial::global_mod = q; // Defines default GF(q)
  ZZ_p::init(q); // Defines GF(q)
  Polynomial phi;
  phi.set_mod(Polynomial::global_mod);



  // for(int d = 1024;d <= 4096;d *= 2){
  for(int d = 4096;d <= 4096;d *= 2){

    std::cout << "d: " << d << std::endl;

    clock_gettime( CLOCK_REALTIME, &start);
    Polynomial::BuildNthCyclotomic(&phi, d); // generate an cyclotomic polynomial
    clock_gettime( CLOCK_REALTIME, &stop);
    for(int i = 0; i <= phi.deg();i++)
      NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
    ZZ_pE::init(NTL_Phi);
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
    CUDAFunctions::init(2*d);

    Polynomial a;
    Polynomial b;
    ///////////////////////////////////////////////
    
    std::cout << "Starting!" <<std::endl;

    // /////////////////////////////////////////////
    // update_crt_spacing
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      Polynomial p;
      p.update_crt_spacing(d);
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "update_crt_spacing: " << diff << " ms" << std::endl;
    crt << d << " " << diff  << std::endl;

    // update_device_data
    Polynomial::random(&a,d-1);
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      a.update_device_data();
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
      a.set_crt_residues_computed(false);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "update_device_data: " << diff << " ms" << std::endl;
    crt << d << " " << diff  << std::endl;

        // update_device_data
    Polynomial::random(&a,d-1);
    a.update_device_data();
    a.set_host_updated(false);
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      a.update_host_data();
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
      a.set_host_updated(false);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "update_host_data: " << diff << " ms" << std::endl;
    crt << d << " " << diff  << std::endl;

    // /////////////////////////////////////////////
    // CRT/ICRT
    
    clock_gettime( CLOCK_REALTIME, &start);
    a.update_host_data();
    a.set_crt_computed(false);
    a.set_crt_residues_computed(false);
    for(int i = 0; i < N;i++){
      a.set_crt_computed(false);
      a.crt();
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "CRT) Foward: " << diff << " ms" << std::endl;
    crt << d << " " << diff  << std::endl;

    cudaMemGetInfo(&f, &t);
    // cout << "Free memory: " << f/(1024*1024) << std::endl;
    // std::cout << "Press Enter to Continue" << std::endl;
    //cin.ignore();

    clock_gettime( CLOCK_REALTIME, &start);
    a.update_host_data();
    for(int i = 0; i < N;i++){
      a.set_icrt_computed(false);
      a.icrt();
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "CRT) Inverse: " << diff << " ms" << std::endl;
    icrt << d << " " << diff  << std::endl;

    // cudaMemGetInfo(&f, &t);
    // cout << "Free memory: " << f/(1024*1024) << std::endl;
    // std::cout << "Press Enter to Continue" << std::endl;
    // cin.ignore();

    clock_gettime( CLOCK_REALTIME, &start);
    Polynomial::random(&a,d-1);
    a.update_device_data();
    for(int i = 0; i < N;i++){
      a.reduce();
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "Reduce) GPU: " << diff << " ms" << std::endl;
    gpu_reduce << d << " " << diff  << std::endl;

    cudaMemGetInfo(&f, &t);
    // cout << "Free memory: " << f/(1024*1024) << std::endl;
    // std::cout << "Press Enter to Continue" << std::endl;
    //cin.ignore();

    clock_gettime( CLOCK_REALTIME, &start);
    a.update_host_data();
    a.set_crt_residues_computed(false);
    for(int i = 0; i < N;i++){
      a.reduce();
      result = cudaDeviceSynchronize();
      a.set_crt_residues_computed(false);
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "Reduce) CPU: " << diff << " ms" << std::endl;
    cpu_reduce << d << " " << diff  << std::endl;

    cudaMemGetInfo(&f, &t);
    // cout << "Free memory: " << f/(1024*1024) << std::endl;
    // std::cout << "Press Enter to Continue" << std::endl;
    // cin.ignore();

    clock_gettime( CLOCK_REALTIME, &start);
    a.update_host_data();
    a.set_crt_residues_computed(false);
    for(int i = 0; i < N;i++){
      a %= q;
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "%q): " << diff << " ms" << std::endl;
    modq << d << " " << diff  << std::endl;

    cudaMemGetInfo(&f, &t);
    // cout << "Free memory: " << f/(1024*1024) << std::endl;
    // std::cout << "Press Enter to Continue" << std::endl;
    //cin.ignore();

    
    ///////////////////////////////////////////////
    // ADD
    
    ////////////////////////////////////
    // Time measured with memory copy //
    ////////////////////////////////////
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);
    // std::cout << a.to_string() << std::endl;       
 
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){

      Polynomial c = (a+b);
      a.set_crt_residues_computed(false);
      b.set_crt_residues_computed(false);
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "ADD) Time measured with memory copy: " << diff << " ms" << std::endl;
    gpu_add_with_memcopy << d << " " << diff  << std::endl;

    ///////////////////////////////////////
    // Time measured without memory copy //
    ///////////////////////////////////////
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);
    
        cudaMemGetInfo(&f, &t);

    // cout << "Free memory: " << f/(1024*1024) << std::endl;
    // std::cout << "Press Enter to Continue" << std::endl;
    //cin.ignore();

    a.update_device_data();
    b.update_device_data();

    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      Polynomial c = (a+b);
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "ADD) Time measured without memory copy: " << diff << " ms" << std::endl;
    gpu_add_without_memcopy << d << " " << diff  << std::endl;


    // ///////////////////////////////////////////////
    // // MUL
    // // Time measured with memory copy
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);
    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){

      Polynomial c = (a*b);
      a.set_crt_residues_computed(false);
      b.set_crt_residues_computed(false);
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "MUL) Time measured with memory copy: " << diff << " ms" << std::endl;
    gpu_mult_with_memcopy << d << " " << diff  << std::endl;
    
    //     cudaMemGetInfo(&f, &t);

    // cout << "Free memory: " << f/(1024*1024) << std::endl;
    // // std::cout << "Press Enter to Continue" << std::endl;
    // //cin.ignore();

    // Time measured without memory copy
    Polynomial::random(&a,d-1);
    Polynomial::random(&b,d-1);
    a.update_crt_spacing(2*d);
    a.update_device_data();
    b.update_crt_spacing(2*d);
    b.update_device_data();

    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < N;i++){
      Polynomial c = (a*b);
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    diff = compute_time_ms(start,stop)/N;
    std::cout << "MUL) Time measured without memory copy: " << diff << " ms" << std::endl;
    gpu_mult_without_memcopy << d << " " << diff  << std::endl;

    cudaMemGetInfo(&f, &t);
    cout << "Used memory: " << (t-f)/(1024*1024) << std::endl;
    cout << "Free memory: " << f/(1024*1024) << std::endl;
    // // std::cout << "Press Enter to Continue" << std::endl;
    // //cin.ignore();

    
  }

  cudaDeviceReset();

}
