#include <assert.h>
#include "distribution.h"

Polynomial Distribution::get_sample(int degree){
  Polynomial p;
  p.update_crt_spacing(degree+1);
  int mod;
  int rec;
  // int phase = 0;
  switch(this->kind){
    case DISCRETE_GAUSSIAN:
      mod = 7;
      rec = 49;
    break;
    case BINARY:
      mod = 2;
      rec = 4;
    break;
    case NARROW:
      mod = 2;
      // phase = 1;
      rec = 4;
    break;
    default:
      mod = 100;
      rec = 10000;
    break;
  }

  // for(int i = 0; i <= degree; i++)
  //   p.set_coeff(i,(rand() % mod - mod/2) - phase);
  // p %= p.get_mod();

  //////////////////////////////////
  // Generate random coefficients //
  //////////////////////////////////
  /**
  * This function supposses all memory used by h_bn_coefs.dp was allocated in 
  * a single cudaMalloc call 
  */
 // if(kind != DISCRETE_GAUSSIAN){
  callCuGetUniformSample(p.h_bn_coefs[0].dp, degree+1);
  cudaError_t result = cudaDeviceSynchronize();// cuRAND doesn't use the same synchronization mechanism as others CUDAs APIs
  assert(result == cudaSuccess);
  
  ///////////////////////////////////////
  // Adjust to the used distribution   //
  ///////////////////////////////////////
  p %= mod;
  p -= mod/2;  
 // }else{
 //  callCuGetNormalSample(p.h_bn_coefs[0].dp, degree+1, 0,(float)(3.1915382432114616));
 //  cudaError_t result = cudaDeviceSynchronize();// cuRAND doesn't use the same synchronization mechanism as others CUDAs APIs
 //  assert(result == cudaSuccess);
 // }

  return p;
}
