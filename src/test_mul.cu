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
  ZZ_pX NTL_Phi;

  ////////////////
  // First test //
  ////////////////
  std::cout << "First test: " << std::endl;
  degree = 8;

  Polynomial::global_mod = conv<ZZ>("1171313591017775093490277364417"); // Defines default GF(q)
  Polynomial::BuildNthCyclotomic(&phi,degree);
  phi.set_mod(Polynomial::global_mod);
  Polynomial::global_phi = &phi;
// 
  srand (36251);

  Polynomial::gen_crt_primes(Polynomial::global_mod,degree);
  std::cout << "M = " << Polynomial::CRTProduct << std::endl;
  ZZ_p::init(Polynomial::global_mod);
  for(int i = 0; i <= phi.deg();i++){
    NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
  }
  ZZ_pE::init(NTL_Phi);

  Polynomial a,b,c;
  Polynomial::random(&a,8);
  Polynomial::random(&b,8);

  c = a*b;
  
  std::cout << "a: " << a.to_string() << std::endl;
  std::cout << "b: " << b.to_string() << std::endl;
  std::cout << "a*b: " <<c.to_string() << std::endl;

  /////////////////
  // Second test //
  /////////////////
  std::cout << std::endl << "Sedcond test: " << std::endl;
  a.set_coeffs(8);
  a.set_coeff(0,1304768405);
  a.set_coeff(1,1912295543);
  a.set_coeff(2,584306160);
  a.set_coeff(3,476779113);
  a.set_coeff(4,2057832156);
  a.set_coeff(5,1274012458);
  a.set_coeff(6,1818971124);
  a.set_coeff(7,1943566066);
  
  b.set_coeffs(8);
  b.set_coeff(0,to_ZZ("13860721405712617825882419437"));
  b.set_coeff(1,to_ZZ("911979968073012785910226533"));
  b.set_coeff(2,to_ZZ("60134818568832314735148266017"));
  b.set_coeff(3,to_ZZ("30228208549061091796489949751"));
  b.set_coeff(4,to_ZZ("42496336825615196107672318132"));
  b.set_coeff(5,to_ZZ("2538357487121433195951809389"));
  b.set_coeff(6,to_ZZ("19401757472799351763805908821"));
  b.set_coeff(7,to_ZZ("68415359680988770693366249506"));

  c.set_coeffs();
  c = a*b;
  
  std::cout << "a: " << a.to_string() << std::endl;
  std::cout << "b: " << b.to_string() << std::endl;
  std::cout << "a*b: " <<c.to_string() << std::endl;
}
