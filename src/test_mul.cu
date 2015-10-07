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

  degree = 4;

  // Polynomial::global_mod = conv<ZZ>("1171313591017775093490277364417"); // Defines default GF(q)
  // Polynomial::global_mod = conv<ZZ>("61"); // Defines default GF(q)
  Polynomial::global_mod = conv<ZZ>("655615111"); // Defines default GF(q)
  Polynomial::BuildNthCyclotomic(&phi,degree);
  // std::cout << phi.to_string() << std::endl;
  phi.set_mod(Polynomial::global_mod);
  Polynomial::global_phi = &phi;

  srand (36251);

  ZZ_p::init(Polynomial::global_mod);
  ZZ_pX NTL_Phi;
  for(int i = 0; i <= phi.deg();i++){
    NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
  }
  ZZ_pE::init(NTL_Phi);

  Polynomial::gen_crt_primes(Polynomial::global_mod,degree);


  Polynomial a,b;
  a.set_device_updated(false);
  b.set_device_updated(false);
  a.set_coeff(0,108142699);
  a.set_coeff(1,162214048);
  a.set_coeff(2,243321072);
  a.set_coeff(3,364981608);
  b.set_coeff(0,1);
  b.set_coeff(1,1);
  b.set_coeff(2,655615110);
  b.set_coeff(3,1);
  a.set_host_updated(true);
  b.set_host_updated(true);

  ZZ_pEX b_ntl;
  ZZ_pEX a_ntl;
  for(int i = 0;i <= a.deg();i++)
    NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
  for(int i = 0;i <= b.deg();i++)
    NTL::SetCoeff(b_ntl,i,conv<ZZ_p>(b.get_coeff(i)));

  Polynomial c = a*b;
  c.icrt();

  ZZ_pEX c_ntl = a_ntl*b_ntl;

  std::cout << "a: " << a.to_string() << " degree: " << a.deg() << std::endl << std::endl;
  std::cout << "b: " << b.to_string() << " degree: " << b.deg() << std::endl << std::endl;
  std::cout << "c: " << c.to_string() << " degree: " << c.deg() << std::endl << std::endl;
  std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
}
