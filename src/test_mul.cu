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

  degree = 2048;

  Polynomial::global_mod = conv<ZZ>("1171313591017775093490277364417"); // Defines default GF(q)
  Polynomial::BuildNthCyclotomic(&phi,degree);
  phi.set_mod(Polynomial::global_mod);
  Polynomial::global_phi = &phi;

  srand (36251);

  Polynomial::gen_crt_primes(Polynomial::global_mod,degree);
  
  ZZ_p::init(Polynomial::global_mod);
  for(int i = 0; i <= phi.deg();i++){
    NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
  }
  ZZ_pE::init(NTL_Phi);

  
  Polynomial a,b;
  Polynomial::random(&a,Polynomial::global_phi->deg()-1);
  Polynomial::random(&b,Polynomial::global_phi->deg()-1);
  Polynomial c = a*b;

  uint64_t start,end;
  std::cout << std::endl << std::endl;
  start = get_cycles();
  for(unsigned int i = 0; i < 10; i++)
    c = a*b;
  end = get_cycles();
  std::cout << "Mul Cycles: " << (end-start)/10 << std::endl;

  ZZ value = to_ZZ(439881456);
  start = get_cycles();
  for(unsigned int i = 0; i < 10; i++)
    Polynomial mod_poly = c%(Polynomial::global_mod);
  end = get_cycles();

  std::cout << "%ZZ Cycles: " << (end-start)/10 << std::endl;

}
