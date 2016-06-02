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
#include <omp.h>
#include <assert.h>
#include <string.h>
#include "polynomial.h"
#include "settings.h"
#include "common.h"
#include "cuda_bn.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cufft.h>

ZZ Polynomial::CRTProduct = ZZ(1);
std::vector<cuyasheint_t> Polynomial::CRTPrimes(0);
std::vector<ZZ> Polynomial::CRTMpi;
std::vector<cuyasheint_t> Polynomial::CRTInvMpi;
ZZ Polynomial::global_mod = ZZ(0);
Polynomial *(Polynomial::global_phi) = NULL;
bool Polynomial::phi_set = false;
std::map<ZZ, std::pair<cuyasheint_t*,int>> reciprocals;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                                                                                                                                                             //
// 29 bits                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        //                                                                                                                                                                                                                                                                                                             //
#if CRTPRIMESIZE == 29
  const uint32_t PRIMES_BUCKET[] = {536870909, 536870879, 536870869, 536870849, 536870839, 536870837, 536870819, 536870813, 536870791, 536870779, 536870767, 536870743, 536870729, 536870723, 536870717, 536870701, 536870683, 536870657, 536870641, 536870627, 536870611, 536870603, 536870599, 536870573, 536870569, 536870563, 536870561, 536870513, 536870501, 536870497, 536870473, 536870401, 536870363, 536870317, 536870303, 536870297, 536870273, 536870267, 536870239, 536870233, 536870219, 536870171, 536870167, 536870153, 536870123, 536870063, 536870057, 536870041, 536870027, 536869999, 536869951, 536869943, 536869937, 536869919, 536869901, 536869891, 536869831, 536869829, 536869793, 536869787, 536869777, 536869771, 536869769, 536869747, 536869693, 536869679, 536869651, 536869637, 536869633, 536869631, 536869607, 536869603, 536869589, 536869583, 536869573, 536869559, 536869549, 536869523, 536869483, 536869471, 536869447, 536869423, 536869409, 536869387, 536869331, 536869283, 536869247, 536869217, 536869189, 536869159, 536869153, 536869117, 536869097, 536869043, 536868979, 536868977, 536868973, 536868961, 536868953, 536868901}; //                                                                                                                                                                                                                                                                                                             //
#else
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                                                                                                                                                             //
  // 15 bits
  #if CRTPRIMESIZE == 15
   const uint32_t PRIMES_BUCKET[] = {32749, 32719, 32717, 32713, 32707, 32693, 32687, 32653, 32647, 32633, 32621, 32611, 32609, 32603, 32587, 32579, 32573, 32569, 32563, 32561, 32537, 32533, 32531, 32507, 32503, 32497, 32491, 32479, 32467, 32443, 32441, 32429, 32423, 32413, 32411, 32401, 32381, 32377, 32371, 32369, 32363, 32359, 32353, 32341, 32327, 32323, 32321, 32309, 32303, 32299, 32297, 32261, 32257, 32251, 32237, 32233, 32213, 32203, 32191, 32189, 32183, 32173, 32159, 32143, 32141, 32119, 32117, 32099, 32089, 32083, 32077, 32069, 32063, 32059, 32057, 32051, 32029, 32027, 32009, 32003, 31991, 31981, 31973, 31963, 31957, 31907, 31891, 31883, 31873, 31859, 31849, 31847, 31817, 31799, 31793, 31771, 31769, 31751, 31741, 31729, 31727, 31723, 31721, 31699, 31687, 31667, 31663, 31657, 31649, 31643, 31627, 31607, 31601, 31583, 31573, 31567, 31547, 31543, 31541, 31531, 31517, 31513, 31511, 31489, 31481, 31477, 31469, 31397, 31393, 31391, 31387, 31379, 31357, 31337, 31333, 31327, 31321, 31319, 31307, 31277, 31271, 31267, 31259, 31253, 31249, 31247, 31237, 31231, 31223, 31219, 31193, 31189, 31183, 31181, 31177, 31159, 31153, 31151, 31147, 31139, 31123, 31121, 31091, 31081, 31079, 31069, 31063, 31051, 31039, 31033, 31019, 31013, 30983, 30977, 30971, 30949, 30941, 30937, 30931, 30911, 30893, 30881, 30871, 30869, 30859, 30853, 30851, 30841, 30839, 30829, 30817, 30809, 30803, 30781, 30773, 30763, 30757, 30727, 30713, 30707}; // //
  #else
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 10 bits
    #if CRTPRIMESIZE == 10
     const uint32_t PRIMES_BUCKET[] = {1021, 1019, 1013, 1009, 997, 991, 983, 977, 971, 967, 953, 947, 941, 937, 929, 919, 911, 907, 887, 883, 881, 877, 863, 859, 857, 853, 839, 829, 827, 823, 821, 811, 809, 797, 787, 773, 769, 761, 757, 751, 743, 739, 733, 727, 719, 709, 701, 691, 683, 677, 673, 661, 659, 653, 647, 643, 641, 631, 619, 617, 613, 607, 601, 599, 593, 587, 577, 571, 569, 563, 557, 547, 541, 523, 521}; ///
    #else
      #if CRTPRIMESIZE == 9
        const uint32_t PRIMES_BUCKET[] = {67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509};
      #endif
    #endif
  #endif
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Polynomial Polynomial::operator+(Polynomial &b){
  Polynomial p = common_addition<Polynomial>(this,&b);
  
  cudaError_t result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);
  
  return p;
}

Polynomial Polynomial::operator+=(Polynomial &b){
  common_addition_inplace<Polynomial>(this,&b);
  
  cudaError_t result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);

  return *this;
}

Polynomial Polynomial::operator*(Polynomial &b){
  Polynomial *p = common_multiplication<Polynomial>(this,&b);

  cudaError_t result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);

  return p;
}

Polynomial Polynomial::operator*=(Polynomial &b){
  common_multiplication_inplace<Polynomial>(this,&b);

  cudaError_t result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);
  return *this;
}

void Polynomial::operator delete(void *ptr){
  Polynomial *p = (Polynomial*)ptr;

  if(p->get_device_crt_residues() != 0x0){
    try
    {
      // std::cout << "Delete: cudaFree" << std::endl;
      cudaError_t result = cudaFree(p->get_device_crt_residues());
      if(result != cudaSuccess)
        throw string( cudaGetErrorString(result));
      
    }catch(string s){
      #ifdef SPEEDCHECK
      std::cout << "Exception at cudaFree: " << s << std::endl;
      #endif 
      cudaGetLastError();//Reset last error
    }
  }

  // free(ptr);
}

void Polynomial::copy_device_crt_residues(Polynomial &b){
  // uint64_t start,end;

  // start = get_cycles();
  if(!b.get_crt_computed())
    return;

  // std::cout << "Will copy residues on device memory" << std::endl;


  // Adjusts CRTSPACING
  // Here we don't use update_crt_spacing(). The reason
  // is: speed and context.
  // update_crt_spacing() may not just update the spacing, but
  // also update device data. And in this context, we know
  // that there is no relevant data to update.
  cuyasheint_t *aux;
  this->CRTSPACING = b.get_crt_spacing();

  cudaError_t result = cudaMalloc((void**)&aux,
                      this->get_crt_spacing()*(Polynomial::CRTPrimes.size())*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);
  result = cudaMemcpyAsync(aux,
                      b.get_device_crt_residues(),
                      this->get_crt_spacing()*(Polynomial::CRTPrimes.size())*sizeof(cuyasheint_t),
                      cudaMemcpyDeviceToDevice);      
  assert(result == cudaSuccess);

  this->set_device_crt_residues(aux);
  // end = get_cycles();
  // std::cout << (end-start) << " cycles" << std::endl;
}
/**
 * Initiates a new bn_t object
 * @param a input: operand
 */

////////////////////////
// Auxiliar functions //
////////////////////////
/**
 * get_words converts a NTL big integer
 * in our bn_t format
 * @param b output: word representation
 * @param a input: operand
 */
void get_words(bn_t *b,ZZ a){
  /**
   * Compute
   */
  cuyasheint_t *h_dp;
  int used = 0;
  int alloc = STD_BNT_WORDS_ALLOC;
  h_dp = (cuyasheint_t *) calloc (alloc,sizeof(cuyasheint_t));

  for(ZZ x = NTL::abs(a); x > 0; x=(x>>WORD),used++){
    if(used >= alloc){
      h_dp = (cuyasheint_t*)realloc(h_dp,alloc+STD_BNT_WORDS_ALLOC);
      alloc += STD_BNT_WORDS_ALLOC;
      std::cout << "get_words realloc! This shouldn't happen." << std::endl;
    }
    h_dp[used] = conv<uint64_t>(x);
  }

  // if(b->alloc != alloc && alloc > 0){
    cudaError_t result;
    if(b->alloc != alloc || b->dp == 0x0){
      // If b->dp was allocated with less data than we need
      if(b->alloc != 0){
        result = cudaFree(b->dp);
        assert(result == cudaSuccess); 
      }
    
      result = cudaMalloc((void**)&b->dp,alloc*sizeof(cuyasheint_t));
      assert(result == cudaSuccess);  
    }

    result = cudaMemcpy(b->dp,h_dp,alloc*sizeof(cuyasheint_t),cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

  // }
  /*
   * Copy new data to device memory
   */

  b->used = used;
  b->alloc = alloc;
  b->sign = (a>=0?BN_POS:BN_NEG);

  free(h_dp);
}
void get_words_host(bn_t *b,ZZ a){
  /**
   * Compute
   */
  cuyasheint_t *h_dp;
  int used = 0;
  int alloc = STD_BNT_WORDS_ALLOC;
  h_dp = (cuyasheint_t *) calloc (alloc,sizeof(cuyasheint_t));

  for(ZZ x = NTL::abs(a); x > 0; x=(x>>WORD),used++){
    if(used >= alloc){
      h_dp = (cuyasheint_t*)realloc(h_dp,alloc+STD_BNT_WORDS_ALLOC);
      alloc += STD_BNT_WORDS_ALLOC;
      std::cout << "get_words realloc!" << std::endl;
    }
    h_dp[used] = conv<uint64_t>(x);
  }

  // if(b->alloc != alloc && alloc > 0){
    // cudaError_t result;
    if(b->alloc != alloc || b->dp == 0x0)
      // If b->dp was allocated with less data than we need
      if(b->alloc != 0)
        free(b->dp);

  /*
   * Copy new data to device memory
   */

  b->used = used;
  b->alloc = alloc;
  b->sign = (a>=0?BN_POS:BN_NEG);
  b->dp = h_dp;
}
void get_words_allocatted(bn_t *b,ZZ a,cuyasheint_t *h_data, cuyasheint_t *d_data, int index,cudaStream_t stream){
  /**
   * Compute
   */
  cuyasheint_t *h_dp;
  int used = 0;
  int alloc = STD_BNT_WORDS_ALLOC;
  h_dp = h_data + index*alloc;

  for(ZZ x = NTL::abs(a); x > 0; x=(x>>WORD),used++){
    assert(used < alloc);
    h_dp[used] = conv<uint64_t>(x);
  }

  /*
   * Copy new data to device memory
   */

  b->used = used;
  b->alloc = alloc;
  b->sign = (a>=0?BN_POS:BN_NEG);
}

int get_used_index(bn_t *coefs,int alloc){
  int i;
  for(i = alloc-1; i >= 0; i--)
    if(coefs[i].used != 0)
      return i;
  return i;
}

/**
 * Convert an array of words into a NTL ZZ
 * @param  a input:array of words
 * @return   output: NTL ZZ
 */
ZZ get_ZZ(bn_t *a){
  ZZ b = conv<ZZ>(0);
  for(int i = a->used-1; i >= 0;i--)
      b = (b<<WORD) | to_ZZ(a->dp[i]);
  return b;
}

__global__ void nothing_here(Complex *a){
  a[0] = a[0];
}

/**
 * Applies NTT
 */
void Polynomial::transf(){
  if(!get_crt_computed())
    crt();
  int N = (get_crt_spacing());

  #ifdef NTTMUL_TRANSFORM

  set_device_crt_residues(
    CUDAFunctions::applyNTT( get_device_crt_residues(), N, CRTPrimes.size(), FORWARD, get_stream() )
  );

  #else
  int size = N*CRTPrimes.size();

  CUDAFunctions::executeCopyIntegerToComplex(d_polyTransf,d_polyCRT,size,get_stream());
  assert(cudaGetLastError() == cudaSuccess);

  cufftExecZ2Z( CUDAFunctions::plan,
                (cufftDoubleComplex *)(d_polyTransf),
                (cufftDoubleComplex *)(d_polyTransf),
                CUFFT_FORWARD
              );
  // assert(fftResult == CUFFT_SUCCESS);

  #endif

  set_transf_computed(true);
  set_crt_computed(false); // I don't remember why this is needed
}

// Applies FFT
void Polynomial::itransf(){
  if(!get_transf_computed())
    return;
  int N = get_crt_spacing();

  #ifdef NTTMUL_TRANSFORM
  // std::cout << "Inverse Transform" << std::endl;
  // int size = N*Polynomial::CRTPrimes.size();

  // cudaError_t result = cudaMemsetAsync(CUDAFunctions::d_mulAux,
  //                                       0,
  //                                       size*sizeof(cuyasheint_t),
  //                                       get_stream());  
  // assert(result == cudaSuccess);

  set_device_crt_residues(
    CUDAFunctions::applyNTT( get_device_crt_residues(), N, CRTPrimes.size(), INVERSE, get_stream())
  );

  #else
  int size = N*CRTPrimes.size();
  
  cufftExecZ2Z( CUDAFunctions::plan,
                (cufftDoubleComplex *)(d_polyTransf),
                (cufftDoubleComplex *)(d_polyTransf),
                CUFFT_INVERSE
              );
  // assert(fftResult == CUFFT_SUCCESS);

  CUDAFunctions::executeCopyAndNormalizeComplexRealPartToInteger(d_polyCRT,(cufftDoubleComplex *)d_polyTransf,size,N,get_stream());
  assert(cudaGetLastError() == cudaSuccess);
  #endif
  set_transf_computed(false); // This is in-place
  set_itransf_computed(true);
}

void Polynomial::crt(){
  if(get_crt_computed())
    return;

  /**
   * To run CRT we need a copy of the coeficients on device's memory
   */
  if(!get_icrt_computed()){
    update_device_data();
    return;
  }

  #ifdef SPEEDCHECK
  std::cout << "Applying CRT." << std::endl;
  #endif
  callCRT(d_bn_coefs,
          (deg()+1),
          get_device_crt_residues(),
          get_crt_spacing(),
          Polynomial::CRTPrimes.size(),
          get_stream()
    );

    // CUDAFunctions::applyNTT(this->get_device_crt_residues(),
                          // this->get_crt_spacing(),
                          // Polynomial::CRTPrimes.size(),
                          // INVERSE,
                          // get_stream());
  set_crt_computed(true);
  
  // NTT-FFT
  transf();
}

void Polynomial::update_device_data(){
  /**
   * This function copy the polynomial to GPU's memory in bn_t format
   */

  if(this->get_crt_computed())
    return;


  this->ON_COPY = true;

  cuyasheint_t *h_data = 0x0;
  if(!get_icrt_computed()){
    #ifdef SPEEDCHECK
    std::cout << "Copying data to GPU." << std::endl;
    #endif
    
    // Verifica se o espaçamento é válido. Se não for, ajusta.
    if(this->get_crt_spacing() < 2*(this->deg()+1)){
      int new_spacing;
      // if((this->deg()+1) > 0)
        // new_spacing = 2*(this->deg()+1);
      // else
        new_spacing = 2*Polynomial::global_phi->deg();

      // Data on device isn't updated (we check it on begginning)
      // So, update_crt_spacing(int) will only update CRTSpacing and alloc memory
      this->update_crt_spacing(new_spacing);
    }

    /**
    * Converts ZZs to bn_t
    */
    h_data = (cuyasheint_t*)calloc(get_crt_spacing()*STD_BNT_WORDS_ALLOC,sizeof(cuyasheint_t));
    // h_data = (cuyasheint_t*)malloc(get_crt_spacing()*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));
    cuyasheint_t *d_data;
    cudaError_t result = cudaMalloc((void**)&d_data,get_crt_spacing()*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

    result = cudaMemsetAsync(d_data,0,get_crt_spacing()*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t),get_stream());
    assert(result == cudaSuccess);

    for(int i = 0; i < (deg()+1); i++)
      get_words_allocatted(&h_bn_coefs[i],get_coeff(i),h_data,d_data,i,get_stream());  

    result = cudaMemcpyAsync(d_data,h_data,get_crt_spacing()*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t),cudaMemcpyHostToDevice,get_stream());
    assert(result == cudaSuccess);
    for(int i = 0; i < get_crt_spacing(); i++)
      h_bn_coefs[i].dp = &d_data[i*STD_BNT_WORDS_ALLOC];

    ////////////////////
    // Copy to device //
    ////////////////////
    result = cudaMemcpyAsync(	d_bn_coefs,
                              h_bn_coefs,
                             get_crt_spacing()*sizeof(bn_t),
                              cudaMemcpyHostToDevice,
                              get_stream() 
                        		);
    assert(result == cudaSuccess);

    set_icrt_computed(true);
  }

  /**
  *  CRT
  */
  this->crt();

  /**
   * Releases memory
   */
  if(h_data)
    free(h_data);
  h_data = 0x0;
  
  this->ON_COPY = false;
  this->set_crt_computed(true);

}

void Polynomial::icrt(){  
  if(get_icrt_computed())
    return;

  // INTT-IFFT
  itransf();

  /**
   * If there is no residue computed on GPU, just copy host data
   */
  // if(!get_itransf_computed()){
    // ();
    // return;
  // }

  #ifdef SPEEDCHECK
  std::cout << "Applying ICRT." << std::endl;
  #endif

  if(get_crt_spacing() > CUDAFunctions::N)
    CUDAFunctions::init(get_crt_spacing());
  
  // CUDAFunctions::applyNTT(this->get_device_crt_residues(),
                          // this->get_crt_spacing(),
                          // Polynomial::CRTPrimes.size(),
                          // INVERSE,
                          // get_stream());

  callICRT( d_bn_coefs,
        get_device_crt_residues(),
        get_crt_spacing(),
        Polynomial::CRTPrimes.size(),
        get_stream()
      );

  this->set_icrt_computed(true);


}

void Polynomial::update_host_data(){
  
    if(get_host_updated())
      return;
    if(!get_icrt_computed())
      this->icrt();
	
    assert(Polynomial::CRTPrimes.size() > 0);
    cudaError_t result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    //////////
    // Copy //
    //////////

    #ifdef SPEEDCHECK
    std::cout << "Copying data to CPU." << std::endl;
    #endif

    assert(get_crt_spacing() > 0);
    result = cudaMemcpy(h_bn_coefs,
                        d_bn_coefs,
                        get_crt_spacing()*sizeof(bn_t),
                        cudaMemcpyDeviceToHost
                        );
    assert(result == cudaSuccess);
    ////////////////////////
    // Convert bn_t to ZZ //
    ////////////////////////
    const int used_coefs = get_used_index(h_bn_coefs,get_crt_spacing())+1;

    /**
     * On update_crt_spacing() we alloc a single continue array of CRTSPACING*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t)
     * positions. So, we don't need to iterate over h_bn_coefs and copy each array. We can just follow the line from 
     * the first element's dp.
     */
    cuyasheint_t *aux = (cuyasheint_t*)malloc(used_coefs*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));
    result = cudaMemcpy(aux,h_bn_coefs[0].dp,h_bn_coefs[0].alloc*used_coefs*sizeof(cuyasheint_t),cudaMemcpyDeviceToHost);
    cudaGetLastError();

    /**
     * We can't use "set_coeffs(int)" here. 
     * It changes CRT_COMPUTED and ICRT_COMPUTED flags.
     */
    this->coefs.clear();
    this->coefs.resize(used_coefs);
    #pragma omp parallel for
    for(int i = 0; i < used_coefs; i++){
      bn_t bn_coef;
      bn_coef.used = h_bn_coefs[i].used;
      bn_coef.alloc = h_bn_coefs[i].alloc;
      bn_coef.sign = h_bn_coefs[i].sign;
      bn_coef.dp = &aux[i*STD_BNT_WORDS_ALLOC];

      ZZ coef = get_ZZ(&bn_coef);
      // ZZ coef = get_ZZ(&bn_coef) % Polynomial::CRTProduct;
      
      if((unsigned int)(i) >= this->coefs.size())
        this->coefs.resize(i+1);
      
      this->coefs[i] = coef;
    }

    free(aux);
    
    this->set_host_updated(true);

    // normalize();
}

void Polynomial::DivRem(Polynomial a,Polynomial b,Polynomial &quot,Polynomial &rem){
  // Returns x = a % b

  if(a.get_host_updated() == false && b.get_host_updated() == false){
    // Operates on GPU
    throw "DivRem for GPU not implemented yet";
  }else{
    if(!a.get_host_updated())
        a.update_host_data();

    if(!b.get_host_updated())
        b.update_host_data();
    
    // No need to reduce
    if(a.deg() <= 0)
      return;
    if(check_special_rem_format(&b)){
      #ifdef VERBOSE
      std::cout << "Rem in special mode."<<std::endl;
      #endif

      const unsigned int half = b.deg()-1;     

      rem.set_coeffs(half+1);
      // #pragma omp parallel for
      for(unsigned int i = 0;i <= half;i++)
        rem.set_coeff(i,a.get_coeff(i)-a.get_coeff(i+half+1));
      rem.set_crt_computed(false);
      rem.set_icrt_computed(false);
      rem.set_host_updated(true);
    }else{
      throw "DivRem: I don't know how to div this!";
    }
  }
}

int isPowerOfTwo (unsigned int x){
  return ((x != 0) && !(x & (x - 1)));
}

void Polynomial::BuildNthCyclotomic(Polynomial *phi,unsigned int n){

  phi->set_coeffs();

  if(isPowerOfTwo(n)){
    #ifdef VERBOSE
    std::cout << n << " is power of 2" << std::endl;
    #endif
    phi->set_coeff(0,1);
    phi->set_coeff(n,1);
    return;
  }else{
    #ifdef VERBOSE
    std::cout << n << " is not power of 2" << std::endl;
    #endif

    std::vector<Polynomial> aux_phi( n+1);

    for (cuyasheint_t i = 1; i <= n; i++) {
       Polynomial t;
       t.set_coeff(0,ZZ(1));

       for (cuyasheint_t j = 1; j <= i-1; j++)
          if (i % j == 0)
             t *= aux_phi[j];

       Polynomial mono;
       mono.set_coeff(i,ZZ(1));
       aux_phi[i] = (mono - 1)/t;

      //  cout << aux_phi[i] << "\n";
    }
    *phi = aux_phi[n];
 }
}

Polynomial Polynomial::get_phi(){
      return *(Polynomial::global_phi);
    }

    /**
     * Computes the reciprocal for an arbitrary ZZ
     * @param b [description]
     */
bn_t get_reciprocal(ZZ q){
      std::pair<cuyasheint_t*,int> pair = reciprocals[q];
      cuyasheint_t *reciprocal = std::get<0>(pair);
      int su = std::get<1>(pair);

      if( reciprocal == NULL){
        /** 
         * Not computed yet
         */
        compute_reciprocal(q);
        pair = reciprocals[q];
        reciprocal = std::get<0>(pair);
        su = std::get<1>(pair);
      }

      bn_t result;
      result.used = su;
      result.alloc = su;
      result.sign = BN_POS;
      result.dp = reciprocal;

      return result;
    }
bn_t get_reciprocal(bn_t q){
	/**
	 * The reciprocal is computed in the first time this function() is called
	 * After that, the result is reused
	 */
	 ZZ q_ZZ = get_ZZ(&q);
      std::pair<cuyasheint_t*,int> pair = reciprocals[q_ZZ];
      cuyasheint_t *reciprocal = std::get<0>(pair);
      int su = std::get<1>(pair);

      if( reciprocal != NULL){
        /** 
         * Not computed yet
         */
        compute_reciprocal(q_ZZ);
        pair = reciprocals[q_ZZ];
        reciprocal = std::get<0>(pair);
        su = std::get<1>(pair);
      }

      bn_t result;
      result.used = su;
      result.alloc = su;
      result.sign = BN_POS;
      result.dp = reciprocal;

      return result;
    }
void compute_reciprocal(ZZ q){
      ZZ u_ZZ;

      // int nwords = NTL::NumBits(q)/WORD + (NTL::NumBits(q)%WORD != 0);
      int nwords = NTL::NumBits(q)/WORD + 1;

      ZZ x = power2_ZZ(2*WORD*nwords);
      u_ZZ = x/q;
      // std::cout << "The reciprocal of " << q << " is " << u_ZZ << std::endl;

      bn_t *h_u;
      h_u = (bn_t*) malloc (sizeof(bn_t));
      h_u->alloc = 0;

      get_words(h_u,u_ZZ);  

      //////////
      // Copy //
      //////////
      cudaError_t result;
      
      // Copy words
      cuyasheint_t *d_u;        
      
      // Copy to device
      result = cudaMalloc((void**)&d_u,h_u->alloc*sizeof(cuyasheint_t));
      assert(result == cudaSuccess);
      
      result = cudaMemcpy(d_u,h_u->dp,h_u->alloc*sizeof(cuyasheint_t),cudaMemcpyHostToDevice);
      assert(result == cudaSuccess);

      reciprocals[q] = std::pair<cuyasheint_t*,int>(d_u,h_u->used);
      
      free(h_u);
    }
