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

  degree = 8;

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

  Polynomial f;
  f.set_coeff(0,to_ZZ("1171313591017775093490277328467"));
  f.set_coeff(1,to_ZZ("1171313591017775093490277328466"));
  f.set_coeff(2,to_ZZ("1171313591017775093490277292515"));
  f.set_coeff(3,to_ZZ("1171313591017775093490277328466"));
  f.set_coeff(4,to_ZZ("1171313591017775093490277328466"));
  f.set_coeff(5,to_ZZ("1171313591017775093490277328466"));
  f.set_coeff(6,to_ZZ("1171313591017775093490277292515"));
  f.set_coeff(7,to_ZZ("1171313591017775093490277328466"));

  Polynomial c;
  c.set_coeff(0,to_ZZ("964667725091769215137108316318"));
  c.set_coeff(1,to_ZZ("1000186797354562661411096296096"));
  c.set_coeff(2,to_ZZ("637227137322806197771212578577"));
  c.set_coeff(3,to_ZZ("724206936004159440113303244679"));
  c.set_coeff(4,to_ZZ("225115344351321797524721056465"));
  c.set_coeff(5,to_ZZ("269275836499117309585514229681"));
  c.set_coeff(6,to_ZZ("346794235475341224013957977025"));
  c.set_coeff(7,to_ZZ("80546323899908571620727771800"));

  Polynomial ff;
  ff = f*f;
  ff.reduce();
  ff %= Polynomial::global_mod;
  std::cout << "f*f: " <<ff.to_string() << std::endl;
  ff.set_device_updated(false);
  
  Polynomial ffc;
  ffc = ff*c;

  std::cout << "ffc: " <<ffc.to_string() << std::endl;
}
