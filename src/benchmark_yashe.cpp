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
#include <fstream>
#include <iterator>
#include <iomanip>
#include "settings.h"
#include "yashe.h"
#include "polynomial.h"
#include "ciphertext.h"
#include <cuda_runtime_api.h>
#include <NTL/ZZ.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>
#include "common.h"

#define BILLION  1000000000L
#define MILLION  1000000L
#define N 2

double runEncrypt(Yashe cipher, int d){
  struct timespec start, stop;
  
  Polynomial a(d);
  a.set_coeff(0,rand());
  a.update_crt_spacing(2*d);
  a.update_device_data();
      
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    cipher.encrypt(a);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
}

double runDecrypt(Yashe cipher, int d){
  struct timespec start, stop;
  
  Polynomial a(2*d-1);
  a.set_coeff(0,rand());
  a.update_device_data();

  Ciphertext c = cipher.encrypt(a);
  c.update_crt_spacing(2*d-1);
  c.update_device_data();

  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    cipher.decrypt(c);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
}

double runAdditionWithMemoryCopy(Yashe cipher, int d){
  struct timespec start, stop;
  
  Polynomial a,b;
  a.set_coeff(0,rand());
  b.set_coeff(0,rand());

  Ciphertext ct_a = cipher.encrypt(a);
  Ciphertext ct_b = cipher.encrypt(b);
  ct_a.update_host_data();
  ct_a.set_crt_computed(false);
  ct_a.set_icrt_computed(false);
  ct_b.update_host_data();
  ct_b.set_crt_computed(false);  
  ct_b.set_icrt_computed(false);  

  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    Ciphertext c= (ct_a+ct_b);

    ct_a.set_crt_computed(false);
    ct_a.set_icrt_computed(false);
    ct_b.set_crt_computed(false);
    ct_b.set_icrt_computed(false);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
}

double runAdditionWithoutMemoryCopy(Yashe cipher, int d){
  struct timespec start, stop;

  Polynomial a,b;
  a.set_coeff(0,rand());
  b.set_coeff(0,rand());
  Ciphertext ct_a = cipher.encrypt(a);
  Ciphertext ct_b = cipher.encrypt(b);
  ct_a.update_device_data();
  ct_b.update_device_data();
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    Ciphertext c = (ct_a+ct_b);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
}

double runMulWithMemoryCopy(Yashe cipher, int d){
  struct timespec start, stop;

  Polynomial a,b;
  a.set_coeff(0,rand());
  b.set_coeff(0,rand());
  Ciphertext ct_a = cipher.encrypt(a);
  Ciphertext ct_b = cipher.encrypt(b);    

  ct_a.update_host_data();
  ct_a.set_crt_computed(false);
  ct_b.update_host_data();
  ct_b.set_crt_computed(false);
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    Ciphertext c =  (ct_a*ct_b);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
}

double runMulWithoutMemoryCopy(Yashe cipher, int d){
  struct timespec start, stop;

  Polynomial a,b;
  a.set_coeff(0,rand());
  b.set_coeff(0,rand());
  Ciphertext ct_a = cipher.encrypt(a);
  Ciphertext ct_b = cipher.encrypt(b);

  ct_a.crt();
  ct_b.crt();
        
  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    Ciphertext c =  (ct_a*ct_b);
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  return compute_time_ms(start,stop)/N;
}

double runKeyswitch(Yashe cipher, int d){
  struct timespec start, stop;

  Polynomial a;
  a.set_coeff(0,rand());
  Ciphertext c = cipher.encrypt(a);
  c.reduce();
  c.update_device_data();

  clock_gettime( CLOCK_REALTIME, &start);
  for(int i = 0; i < N;i++){
    c.convert();
    cudaDeviceSynchronize();
  }
  clock_gettime( CLOCK_REALTIME, &stop);
  
  return compute_time_ms(start,stop)/N;
}

int main(int argc, char* argv[]){
    struct timespec start, stop;
    double diff;

    std::cout << time(NULL) << std::endl;
    
    clock_gettime( CLOCK_REALTIME, &start);
    sleep(1);    
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << time(NULL) << std::endl;

    std::cout << " 1 sec: " << compute_time_ms(stop,start) << std::endl;

    // Polynomial R;
    uint64_t t;
    t = 1024;
    Polynomial phi;
    ZZ_pX NTL_Phi;

    cout << fixed;
    cout.precision(2);
    ZZ q;
    NTL::power2(q,1279);
    // NTL::power2(q,521);
    // NTL::power2(q,127);
    q -= conv<ZZ>("1");
    Polynomial::global_mod = q;
    ZZ_p::init(q); // Defines GF(q)

    uint64_t w = 4294967296;

    Polynomial::BuildNthCyclotomic(&phi, 32); // generate an cyclotomic polynomial
    Polynomial::global_phi = &phi;
    for(int i = 0; i <= phi.deg();i++)
      NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
    ZZ_pE::init(NTL_Phi);


    Polynomial::gen_crt_primes(Polynomial::global_mod,8);
    CUDAFunctions::init(64);// Warming cuda api
    ////////////////////////////////

    ofstream encrypt;
    ofstream decrypt;
    ofstream add_with_memcopy;
    ofstream add_without_memcopy;
    ofstream mult_with_memcopy;
    ofstream mult_without_memcopy;
    ofstream keyswitch;

    std::string encrypt_filename;
    std::string decrypt_filename;
    std::string add_with_memcopy_filename;
    std::string add_withou_memcopy_tfilename;
    std::string mult_with_memcopy_filename;
    std::string mult_without_memcopy_filename;
    std::string keyswitch_filename;

    if(argc == 2){
      char* suffix = argv[1];

      encrypt_filename = "encrypt_"+current_date_time()+"_"+suffix+".dat";
      decrypt_filename = "decrypt_"+current_date_time()+"_"+suffix+".dat";
      add_with_memcopy_filename = "add_with_memcopy_"+current_date_time()+"_"+suffix+".dat";
      add_withou_memcopy_tfilename = "add_withou_tmemcopy_"+current_date_time()+"_"+suffix+".dat";
      mult_with_memcopy_filename = "mult_with_memcopy_"+current_date_time()+"_"+suffix+".dat";
      mult_without_memcopy_filename = "mult_without_memcopy_"+current_date_time()+"_"+suffix+".dat";
      keyswitch_filename = "keyswitch_"+current_date_time()+"_"+suffix+".dat";
    }else{   
      encrypt_filename = "encrypt_"+current_date_time()+".dat";
      decrypt_filename = "decrypt_"+current_date_time()+".dat";
      add_with_memcopy_filename = "add_with_memcopy_"+current_date_time()+".dat";
      add_withou_memcopy_tfilename = "add_withou_tmemcopy_"+current_date_time()+".dat";
      mult_with_memcopy_filename = "mult_with_memcopy_"+current_date_time()+".dat";
      mult_without_memcopy_filename = "mult_without_memcopy_"+current_date_time()+".dat";
      keyswitch_filename = "keyswitch_"+current_date_time()+".dat";
    }
    encrypt.open (encrypt_filename);
    decrypt.open (decrypt_filename);
    add_with_memcopy.open (add_with_memcopy_filename);
    add_without_memcopy.open (add_withou_memcopy_tfilename);
    mult_with_memcopy.open (mult_with_memcopy_filename);
    mult_without_memcopy.open (mult_without_memcopy_filename);
    keyswitch.open (keyswitch_filename);

    std::cout << "Writing encrypt data to " << encrypt_filename << std::endl;
    std::cout << "Writing decrypt data to " << decrypt_filename << std::endl;
    std::cout << "Writing add_with_memcopy data to " << add_with_memcopy_filename << std::endl;
    std::cout << "Writing add_without_memcopy data to " << add_withou_memcopy_tfilename << std::endl;
    std::cout << "Writing mult_with_memcopy data to " << mult_with_memcopy_filename << std::endl;
    std::cout << "Writing mult_without_memcopy data to " << mult_without_memcopy_filename << std::endl;
    std::cout << "Writing keyswitch data to " << keyswitch_filename << std::endl;
    ZZ_p::init(q); // Defines GF(q)
       
    // for(int d = 1024;d <= 8192;d *= 2){
    for(int d = 16384;d <= 16384;d *= 2){

    //////////////////////
    // Polynomial setup //
    //////////////////////
    std::cout << "d: " << d << std::endl;

    Polynomial::BuildNthCyclotomic(&phi, d); // generate an cyclotomic polynomial
           
    std::cout << "Irreducible polynomial generated in " << diff << " ms." << std::endl;
    std::cout << "Generating " << phi.deg() << " degree polynomials." << std::endl;
    phi.set_mod(Polynomial::global_mod);
    Polynomial::global_phi = &phi;

    // Set params to NTL 
    ZZ_p::init(Polynomial::global_mod);
    for(int i = 0; i <= phi.deg();i++){
      NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
    }
    ZZ_pE::init(NTL_Phi);

    Polynomial::gen_crt_primes(Polynomial::global_mod,d);
    CUDAFunctions::init(d);
    
    std::cout << Polynomial::CRTPrimes.size() <<  " CRT primes generated in " << diff << " ms." << std::endl;
    std::cout << "q: " << NTL::NumBytes(q)*8 << " bits" << std::endl;

    //////////////////
    // Cipher setup //
    //////////////////
    Yashe cipher = Yashe();

    Yashe::d = d;
    Yashe::phi = phi;
    Yashe::q = q;
    Yashe::t = t;
    Yashe::w = conv<ZZ>(w);
    Yashe::lwq = floor(NTL::log(q)/NTL::log(to_ZZ(w)))+1;

      
    clock_gettime( CLOCK_REALTIME, &start);
    cipher.generate_keys();
        
    clock_gettime( CLOCK_REALTIME, &stop);
      std::cout << "Keys generated in " << compute_time_ms(start,stop) << " ms." << std::endl;

    ////////////////
    // Get sample //
    ////////////////
    Distribution xkey = Distribution(NARROW);
    Polynomial p(d+1);

    // clock_gettime( CLOCK_REALTIME, &start);
    // for(int i = 0; i < N;i++){
    //   xkey.get_sample(&p,d+1);
    //   cudaDeviceSynchronize();
    // }
    // clock_gettime( CLOCK_REALTIME, &stop);
    // diff = compute_time_ms(start,stop)/N;
    // std::cout << "get_sample) " << diff << " ms" << std::endl;
        

    // diff = runEncrypt(cipher, d);
    // std::cout << "Encrypt) Time measured with memory copy: " << diff << " ms" << std::endl;
    // encrypt << d << " " << diff << std::endl;
    
    //       size_t t,f;
    //   cudaMemGetInfo(&f, &t);
    //   cout << "Used memory: " << (t-f)/(1024*1024) << std::endl;
    //   cout << "Free memory: " << f/(1024*1024) << std::endl;
    // diff = runDecrypt(cipher, d);
    // std::cout << "Decrypt) Time measured with memory copy: " << diff << " ms" << std::endl;
    // decrypt << d << " " << diff << std::endl;;

    //   cudaMemGetInfo(&f, &t);
    //   cout << "Used memory: " << (t-f)/(1024*1024) << std::endl;
    //   cout << "Free memory: " << f/(1024*1024) << std::endl;
    diff = runAdditionWithoutMemoryCopy(cipher, d);
    std::cout << "Homomorphic Addition) Time measured without memory copy: " << diff << " ms" << std::endl;
    add_without_memcopy << d << " " << diff << std::endl;;
    
    diff = runMulWithoutMemoryCopy(cipher, d);
    std::cout << "Homomorphic Multiplication) Time measured without memory copy: " << diff << " ms" << std::endl;
    mult_without_memcopy << d << " " << diff << std::endl;;

    diff = runKeyswitch(cipher, d);
    std::cout << "KeySwitch) Time measured with memory copy: " << diff << " ms" << std::endl;
    keyswitch << d << " " << diff << std::endl;;
  }
  cudaDeviceReset();
}
