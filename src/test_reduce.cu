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

  Polynomial::gen_crt_primes(Polynomial::global_mod,degree);
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

  a.set_device_updated(false);
  a.set_host_updated(true);
  a.reduce();

  std::cout << "CPU: " <<a.to_string() << std::endl;

}
