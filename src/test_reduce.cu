#include <NTL/ZZ_pEX.h>

#include "polynomial.h"
#include "settings.h"
#include "cuda_functions.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

int main(){
  int degree;
  Polynomial phi;

  degree = 8;

  Polynomial::global_mod = conv<ZZ>("655615111"); // Defines default GF(q)
  Polynomial::BuildNthCyclotomic(&phi,degree);
  phi.set_mod(Polynomial::global_mod);
  Polynomial::global_phi = &phi;

  srand (36251);

  // Polynomial::gen_crt_primes(Polynomial::global_mod,degree);

  Polynomial::CRTPrimes.clear();
  Polynomial::CRTMpi.clear()
;  Polynomial::CRTInvMpi.clear();

  Polynomial::CRTPrimes.push_back(751);
  Polynomial::CRTPrimes.push_back(839);
  Polynomial::CRTPrimes.push_back(829);
  Polynomial::CRTPrimes.push_back(661);
  Polynomial::CRTPrimes.push_back(797);
  Polynomial::CRTPrimes.push_back(1019);
  Polynomial::CRTPrimes.push_back(857);
  Polynomial::CRTProduct = to_ZZ("240309652370522267791");

  // Compute M/pi and it's inverse
  for(unsigned int i = 0; i < Polynomial::CRTPrimes.size();i++){
    ZZ pi = to_ZZ(Polynomial::CRTPrimes[i]);
    Polynomial::CRTMpi.push_back(Polynomial::CRTProduct/pi);
    Polynomial::CRTInvMpi.push_back(conv<cuyasheint_t>(NTL::InvMod(Polynomial::CRTMpi[i]%pi,pi)));
  }
  
  CUDAFunctions::write_crt_primes();


  ZZ_p::init(Polynomial::global_mod);

  std::cout << "Phi: " << phi.to_string() << std::endl;

  // GPU
  Polynomial a;
  a.set_coeff(0,1);
  a.set_coeff(1,1);
  a.set_coeff(2,1);
  a.set_coeff(3,1);
  a.set_coeff(4,1);
  a.set_coeff(5,1);
  a.set_coeff(6,1);
  a.set_coeff(7,1);
  a.set_coeff(8,1);

  a.update_device_data();
  a.set_host_updated(false);
  a.reduce();

  std::cout << "GPU: " <<a.to_string() << std::endl;

  // CPU
  a.set_coeff(0,1);
  a.set_coeff(1,1);
  a.set_coeff(2,1);
  a.set_coeff(3,1);
  a.set_coeff(4,1);
  a.set_coeff(5,1);
  a.set_coeff(6,1);
  a.set_coeff(7,1);
  a.set_coeff(8,1);

  a.set_crt_residues_computed(false);
  a.set_host_updated(true);
  a.reduce();

  std::cout << "CPU: " <<a.to_string() << std::endl;


 // GPU
  a.set_coeff(0,722); 
  a.set_coeff(1,173); 
  a.set_coeff(2,735); 
  a.set_coeff(3,651); 
  a.set_coeff(4,460); 
  a.set_coeff(5,161); 
  a.set_coeff(6, 56); 
  a.set_coeff(7, 80); 
  a.set_coeff(8,762); 
  a.set_coeff(9,  9); 
  a.set_coeff(10,220); 
  a.set_coeff(11,281);  
  a.set_coeff(12, 62);  
  a.set_coeff(13, 96);  
  a.set_coeff(14,239);  

  a.update_device_data();
  a.set_host_updated(false);
  a.reduce();
  // a %= Polynomial::global_mod;

  std::cout << "GPU: " <<a.to_string() << std::endl;

  // CPU
  a.set_coeff(0,722); 
  a.set_coeff(1,173); 
  a.set_coeff(2,735); 
  a.set_coeff(3,651); 
  a.set_coeff(4,460); 
  a.set_coeff(5,161); 
  a.set_coeff(6, 56); 
  a.set_coeff(7, 80); 
  a.set_coeff(8,762); 
  a.set_coeff(9,  9); 
  a.set_coeff(10,220); 
  a.set_coeff(11,281);  
  a.set_coeff(12, 62);  
  a.set_coeff(13, 96);  
  a.set_coeff(14,239);  

  a.set_crt_residues_computed(false);
  a.set_host_updated(true);
  a.reduce();
  // a %= Polynomial::global_mod;

  std::cout << "CPU: " <<a.to_string() << std::endl;


}
