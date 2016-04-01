#include <assert.h>
#include "distribution.h"


//////////////
// fastrand //
//////////////
static unsigned int g_seed = SEED;

inline int fastrand(){
  g_seed = (214013*g_seed + 2531011);

  return (g_seed>>16)&0x7FFF;
}

///////////
// Intel //
///////////

/////////////////////////////////////////////////////////////////////////////

// The Software is provided "AS IS" and possibly with faults. 

// Intel disclaims any and all warranties and guarantees, express, implied or

// otherwise, arising, with respect to the software delivered hereunder,

// including but not limited to the warranty of merchantability, the warranty

// of fitness for a particular purpose, and any warranty of non-infringement

// of the intellectual property rights of any third party.

// Intel neither assumes nor authorizes any person to assume for it any other

// liability. Customer will use the software at its own risk. Intel will not

// be liable to customer for any direct or indirect damages incurred in using

// the software. In no event will Intel be liable for loss of profits, loss of

// use, loss of data, business interruption, nor for punitive, incidental,

// consequential, or special damages of any kind, even if advised of

// the possibility of such damages.

//

// Copyright (c) 2003 Intel Corporation

//

// Third-party brands and names are the property of their respective owners

//

///////////////////////////////////////////////////////////////////////////

// Random Number Generation for SSE / SSE2

// Source File

// Version 0.1

// Author Kipp Owens, Rajiv Parikh

////////////////////////////////////////////////////////////////////////



#include <emmintrin.h>

static __m128i cur_seed;

void srand_sse( unsigned int seed )
{
  cur_seed = _mm_set_epi32( seed, seed+1, seed, seed+1 );
}

inline void rand_sse( int  *result )
{
   __attribute__ ((aligned(16))) __m128i cur_seed_split;
   __attribute__ ((aligned(16))) __m128i multiplier;
   __attribute__ ((aligned(16))) __m128i adder;
   __attribute__ ((aligned(16))) __m128i mod_mask;
   __attribute__ ((aligned(16))) __m128i sra_mask;
   __attribute__ ((aligned(16))) __m128i sseresult;
   __attribute__ ((aligned(16))) static const unsigned int mult[4] = { 214013, 17405, 214013, 69069 };
   __attribute__ ((aligned(16))) static const unsigned int gadd[4] = { 2531011, 10395331, 13737667, 1 };
   __attribute__ ((aligned(16))) static const unsigned int mask[4] = { 0xFFFFFFFF, 0, 0xFFFFFFFF, 0 };
   __attribute__ ((aligned(16))) static const unsigned int masklo[4] = { 0x00007FFF, 0x00007FFF, 0x00007FFF, 0x00007FFF };

  adder = _mm_load_si128( (__m128i*) gadd);
  multiplier = _mm_load_si128( (__m128i*) mult);
  mod_mask = _mm_load_si128( (__m128i*) mask);
  sra_mask = _mm_load_si128( (__m128i*) masklo);
  cur_seed_split = _mm_shuffle_epi32( cur_seed, _MM_SHUFFLE( 2, 3, 0, 1 ) );

  cur_seed = _mm_mul_epu32( cur_seed, multiplier );
  multiplier = _mm_shuffle_epi32( multiplier, _MM_SHUFFLE( 2, 3, 0, 1 ) );
  cur_seed_split = _mm_mul_epu32( cur_seed_split, multiplier );

  cur_seed = _mm_and_si128( cur_seed, mod_mask);
  cur_seed_split = _mm_and_si128( cur_seed_split, mod_mask );
  cur_seed_split = _mm_shuffle_epi32( cur_seed_split, _MM_SHUFFLE( 2, 3, 0, 1 ) );
  cur_seed = _mm_or_si128( cur_seed, cur_seed_split );
  cur_seed = _mm_add_epi32( cur_seed, adder);

  sseresult = _mm_srai_epi32( cur_seed, 16);
  sseresult = _mm_and_si128( sseresult, sra_mask );
  _mm_storeu_si128( (__m128i*) result, sseresult );

  return;
}

void Distribution::generate_sample(Polynomial *p,int mod,int degree){
  /////////
  // NTL //
  /////////
  // for(int i = 0; i <= degree; i++){
  //   if(mod == 2){
  //     p.set_coeff(i,RandomLen_ZZ(1)); // random number between 0 and n-1
  //   }else
  //     p.set_coeff(i,RandomBnd(mod));
  // }
   
  //////////
  // RAND //
  //////////
  // for(int i = 0; i <= degree; i++){
  //     p.set_coeff(i,rand()%mod);
  // }

  //////////////
  // fastrand //
  //////////////
  // for(int i = 0; i <= degree; i++){
      // p.set_coeff(i,fastrand()%mod);
  // }

  // for(int i = 0; i <= degree; i+=4){
  //   int a[4];
  //   rand_sse(a);
  //   p.set_coeff(i,a[0]%mod);
  //   p.set_coeff(i+1,a[1]%mod);
  //   p.set_coeff(i+2,a[2]%mod);
  //   p.set_coeff(i+3,a[3]%mod);
  // }
  //////////////////////////////////
  // Generate random coefficients //
  //////////////////////////////////
  /**
  * This function supposses all memory used by h_bn_coefs.dp was allocated in 
  * a single cudaMalloc call 
  */
 // if(kind != DISCRETE_GAUSSIAN){
  // callCuGetUniformSample(p->h_bn_coefs[0].dp, p->d_bn_coefs, degree+1, mod);
  // p->set_icrt_computed(true);
  // p->set_crt_computed(false);
  // p->set_host_updated(false);
  callCuGetUniformSampleCRT(p->get_device_crt_residues(), degree +1,Polynomial::CRTPrimes.size(), mod);
  p->set_icrt_computed(false);
  p->set_crt_computed(true);
  p->set_itransf_computed(false);
  p->set_transf_computed(false);
  p->set_host_updated(false);
  
  ///////////////////////////////////////
  // Adjust to the used distribution   //
  ///////////////////////////////////////
 // }else{
 //  callCuGetNormalSample(p.h_bn_coefs[0].dp, degree+1, 0,(float)(3.1915382432114616));
 //  cudaError_t result = cudaDeviceSynchronize();// cuRAND doesn't use the same synchronization mechanism as others CUDAs APIs
 //  assert(result == cudaSuccess);
 // }

}

Polynomial Distribution::get_sample(int degree,int spacing){
  Polynomial p(spacing);
  int mod;
  // int rec;
  // int phase = 0;
  switch(this->kind){
    case DISCRETE_GAUSSIAN:
      mod = 7;
    break;
    case BINARY:
      mod = 2;
    break;
    case NARROW:
      mod = 2;
    break;
    default:
      mod = 100;
    break;
  }

  p.set_coeffs(degree+1);
  // p.update_crt_spacing(2*(degree+1));
  generate_sample(&p,mod,degree);
  return p;
}


Polynomial Distribution::get_sample(Polynomial *p, int degree){
  int mod;
  int rec;
  // int phase = 0;
  switch(this->kind){
    case DISCRETE_GAUSSIAN:
      mod = 7;
    break;
    case BINARY:
      mod = 2;
    break;
    case NARROW:
      mod = 2;
    break;
    default:
      mod = 100;
    break;
  }


  p->set_icrt_computed(false);
  p->set_crt_computed(false);
  p->set_itransf_computed(false);
  p->set_transf_computed(false);
  p->set_host_updated(true);
  p->update_crt_spacing(2*(degree+1));
  p->set_coeffs(degree+1);
  generate_sample(p,mod,degree);
  return p;
}

Polynomial Distribution::get_sample(int degree){
  return get_sample(degree,degree+1);
}