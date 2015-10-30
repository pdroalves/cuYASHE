#include "distribution.h"

Polynomial Distribution::get_sample(int degree){
  Polynomial p;
  int mod;
  int phase = 0;
  switch(this->kind){
    case DISCRETE_GAUSSIAN:
      mod = 7;
    break;
    case BINARY:
      mod = 2;
    break;
    case NARROW:
      mod = 2;
      phase = 1;
    break;
    default:
      mod = 100;
    break;
  }

  p.set_coeffs(degree+1);
  for(int i = 0; i <= degree; i++)
    p.set_coeff(i,(rand() % mod - mod/2) - phase);
  
  p.update_crt_spacing();
  return p;
}
