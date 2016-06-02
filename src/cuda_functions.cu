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
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "cuda_functions.h"
#include "cuda_bn.h"
#include "settings.h"
#include "polynomial.h"

// #ifdef NTTMUL
// #define PRIMEP (int)2147483647
// #define PRIMITIVE_ROOT (int)7;//2^31-1 fails the test(P-1)%N
// #define PRIMEP (uint32_t)4294955009
// #define PRIMITIVE_ROOT (int)3
#define PRIMEP (uint64_t)18446744069414584321
#define PRIMITIVE_ROOT (int)7
#define W2 (uint64_t)16777216
#define W4 (uint64_t)281474976710656L
ZZ PZZ = to_ZZ(PRIMEP); 

#ifdef NTTMUL_TRANSFORM
  int CUDAFunctions::transform = NTTMUL;
#else
  int CUDAFunctions::transform = CUFFTMUL;
#endif

cuyasheint_t CUDAFunctions::wN = 0;
cuyasheint_t *CUDAFunctions::d_W = NULL;//W and WInv doesn't fit constant memory
cuyasheint_t *CUDAFunctions::d_WInv = NULL;
cuyasheint_t *CUDAFunctions::d_inner_results = NULL;
cuyasheint_t *CUDAFunctions::d_inner_results_used = NULL;
cuyasheint_t *CUDAFunctions::d_mulA = NULL;
cuyasheint_t *CUDAFunctions::d_mulB = NULL;
cuyasheint_t *CUDAFunctions::d_mulAux = NULL;
Complex *CUDAFunctions::d_mulComplexA = NULL;
Complex *CUDAFunctions::d_mulComplexB = NULL;
Complex *CUDAFunctions::d_mulComplexC = NULL;
extern __constant__ cuyasheint_t M[STD_BNT_WORDS_ALLOC];
extern __constant__ int M_used;
extern __constant__ cuyasheint_t u[STD_BNT_WORDS_ALLOC];
extern __constant__ int u_used;
extern __constant__ cuyasheint_t CRTPrimesConstant[PRIMES_BUCKET_SIZE];

// __constant__ cuyasheint_t W16[225]; 
// __constant__ cuyasheint_t WInv16[225]; 
// __constant__ cuyasheint_t W8[50]; 
// __constant__ cuyasheint_t WInv8[50]; 

// #elif defined(CUFFTMUL)
cufftHandle CUDAFunctions::plan;
// #endif
int CUDAFunctions::N = 0;

__host__ __device__ inline  uint64_t s_add(uint64_t a,uint64_t b);
__host__ __device__ inline uint64_t s_sub(uint64_t a,uint64_t b);

static __device__ inline Complex ComplexMul(Complex a, Complex b);
static __device__ inline Complex ComplexAdd(Complex a, Complex b);
static __device__ inline Complex ComplexSub(Complex a, Complex b);


///////////////////////
// Memory operations //
///////////////////////


__global__ void realignCRTResidues(int oldSpacing,int newSpacing, cuyasheint_t *array,cuyasheint_t *new_array,int residuesSize,int residuesQty){
  //
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int residueId = (newSpacing < oldSpacing ? tid / newSpacing: tid / oldSpacing);
  const int new_array_offset = (newSpacing < oldSpacing ? (tid % newSpacing) + residueId*newSpacing:(tid % oldSpacing) + residueId*newSpacing);
  const int old_array_offset = (newSpacing < oldSpacing ? (tid % newSpacing) + residueId*oldSpacing:(tid % oldSpacing) + residueId*oldSpacing);

  if(new_array_offset < newSpacing*residuesQty && old_array_offset < oldSpacing*residuesQty )
    new_array[new_array_offset] = array[old_array_offset];

}


__host__ cuyasheint_t* CUDAFunctions::callRealignCRTResidues(cudaStream_t stream,
                                                              const int oldSpacing,
                                                              const int newSpacing,
                                                              cuyasheint_t *array,
                                                              const int residuesSize,
                                                              const int residuesQty){
  if(oldSpacing == newSpacing)
    return NULL;
  #ifdef VERBOSE
  std::cout << "Realigning..." << std::endl;
  #endif
  
  int size;
  if(newSpacing < oldSpacing)
    size = newSpacing*residuesQty;
  else
    size = oldSpacing *residuesQty;
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  cuyasheint_t *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,newSpacing*residuesQty*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);
  result = cudaMemsetAsync( d_new_array,
                            0,
                            newSpacing*residuesQty*sizeof(cuyasheint_t),
                            stream);
  assert(result == cudaSuccess);

  realignCRTResidues <<< gridDim,blockDim, 0, stream >>> (oldSpacing,newSpacing,array,d_new_array,residuesSize,residuesQty);
  result = cudaGetLastError();
  assert(result == cudaSuccess);

  return d_new_array;
}

///////////////////////////////////////
/// ADD
///////////////////////////////////////

#ifdef NTTMUL_TRANSFORM
__global__ void polynomialAddSub(const int OP,const cuyasheint_t *a,const cuyasheint_t *b,cuyasheint_t *c,const int size,const int N){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      if(OP == ADD){
        c[tid] = s_add(a[tid],b[tid]);
        // if(c[tid] < a[tid])
          // printf("Overflow!");
      }else
        c[tid] = s_sub(a[tid],b[tid]);
  }
}

__host__ void CUDAFunctions::callPolynomialAddSub(cuyasheint_t *c,cuyasheint_t *a,cuyasheint_t *b,int size,int OP,cudaStream_t stream){
  // This method expects that both arrays are aligned
  int nthreads = 128;
  int ADDGRIDXDIM = (size%nthreads == 0? size/nthreads : size/nthreads + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(nthreads);

  polynomialAddSub <<< gridDim,blockDim,0,stream  >>> (OP,a,b,c,size,N);
  assert(cudaGetLastError() == cudaSuccess);
  #ifdef VERBOSE
  std::cout << "polynomialAdd kernel:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
  #endif
}


__global__ void polynomialAddSubInPlace(const int OP, cuyasheint_t *a,const cuyasheint_t *b,const int size,const int N){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  // const int rid = tid / N; // Residue id
  cuyasheint_t a_value;
  cuyasheint_t b_value;

  if(tid < size ){
      // printf("A[0]: %d\nB[0]: %d\n\n",a[tid],b[tid]);
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = a[tid];
      b_value = b[tid];

      if(OP == ADD){
        a_value = s_add(a_value,b_value);
        // if(a_value < a[tid])
          // printf("Overflow!\n");
      }else
        a_value = s_sub(a_value,b_value);

      a[tid] = a_value;
  }
}

__host__ void CUDAFunctions::callPolynomialAddSubInPlace(cudaStream_t stream,cuyasheint_t *a,cuyasheint_t *b,int size,int OP){
  // This method expects that both arrays are aligned
  // Add and store in array a
  int nthreads = 128;
  int ADDGRIDXDIM = (size%nthreads == 0? size/nthreads : size/nthreads + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  polynomialAddSubInPlace <<< gridDim,blockDim,0,stream >>> (OP,a,b,size,N);
  assert(cudaGetLastError() == cudaSuccess);
  #ifdef VERBOSE
  std::cout << "polynomialAdd kernel:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
  #endif
}

#else
__global__ void polynomialcuFFTAddSub(const int OP,const Complex *a,const Complex *b,Complex *c,const int size,const int N){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      if(OP == ADD){
        c[tid] = ComplexAdd(a[tid],b[tid]);
      }else
      c[tid] = ComplexSub(a[tid],b[tid]);
  }
}

__host__ void CUDAFunctions::callPolynomialcuFFTAddSub(Complex *c,Complex *a,Complex *b,int size,int OP,cudaStream_t stream){
  // This method expects that both arrays are aligned
  int nthreads = 64;
  int ADDGRIDXDIM = (size%nthreads == 0? size/nthreads : size/nthreads + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(nthreads);

  polynomialcuFFTAddSub <<< gridDim,blockDim,0,stream  >>> (OP,a,b,c,size,N);
  assert(cudaGetLastError() == cudaSuccess);
  #ifdef VERBOSE
  std::cout << "polynomialAdd kernel:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
  #endif
}


__global__ void polynomialcuFFTAddSubInPlace(const int OP, Complex *a,const Complex *b,const int size,const int N){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  // const int rid = tid / N; // Residue id
  Complex a_value;
  Complex b_value;

  if(tid < size ){
      // printf("A[0]: %d\nB[0]: %d\n\n",a[tid],b[tid]);
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = a[tid];
      b_value = b[tid];

      if(OP == ADD)
        a_value = ComplexAdd(a_value,b_value);
      else
        a_value = ComplexSub(a_value,b_value);

      a[tid] = a_value;
  }
}

__host__ void CUDAFunctions::callPolynomialcuFFTAddSubInPlace(cudaStream_t stream,Complex *a,Complex *b,int size,int OP){
  // This method expects that both arrays are aligned
  // Add and store in array a
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  polynomialcuFFTAddSubInPlace <<< gridDim,blockDim,0,stream >>> (OP,a,b,size,N);
  assert(cudaGetLastError() == cudaSuccess);
  #ifdef VERBOSE
  std::cout << "polynomialAdd kernel:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
  #endif
}
#endif

///////////////////////////////////////

///////////////////////////////////////
/// MUL

// #if defined(CUFFTMUL)

__global__ void copyIntegerToComplex(Complex *a,cuyasheint_t *b,int size){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size ){
      a[tid].x =   __ull2double_rn(b[tid]);
      // printf("%ld => %f\n\n",b[tid],a[tid].x);
      a[tid].y = 0;
  }else{
    // a[tid].x = 0;
    // a[tid].y = 0;
  }
}

__global__ void copyAndRealignIntegerToComplex(Complex *a,cuyasheint_t *b,const unsigned oldSpacing,const unsigned int newSpacing,const unsigned int residuesQty){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int residueId = (newSpacing < oldSpacing ? tid / newSpacing: tid / oldSpacing);
  const int new_array_offset = (newSpacing < oldSpacing ? (tid % newSpacing) + residueId*newSpacing:(tid % oldSpacing) + residueId*newSpacing);
  const int old_array_offset = (newSpacing < oldSpacing ? (tid % newSpacing) + residueId*oldSpacing:(tid % oldSpacing) + residueId*oldSpacing);

  if(new_array_offset < newSpacing*residuesQty && old_array_offset < oldSpacing*residuesQty ){
      a[new_array_offset].x =  __ull2double_rn(b[old_array_offset]);
      a[new_array_offset].y = 0;
  }
}

__global__ void copyAndNormalizeComplexRealPartToInteger(cuyasheint_t *b,const Complex *a,const int size,const int N){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if(tid < size ){
      double scale = 1.0/N;
      b[tid] = rint(a[tid].x*scale);
  }
}
////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Addition
static __device__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}


static __device__ inline Complex ComplexSub(Complex a, Complex b)
{
    Complex c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

// Complex multiplication
static __device__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}


// Complex pointwise multiplication
__global__ void polynomialcuFFTMul(Complex *c, const Complex *a,const Complex *b,int size){
    const int tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid < size  ){
        c[tid] = ComplexMul(a[tid], b[tid]);
    }else{
      c[tid].x = 0;
      c[tid].y = 0;
    }
}
// #elif defined(NTTMUL)

__host__ __device__ bool overflow(const uint64_t a, const uint64_t b){
  // True if a+b will result in a integer overflow.
  return (a+b) < a;
  // return lessThan((a+b),a);
}

__host__ __device__ uint64_t s_rem (uint64_t a)
{
  // Special reduction for prime 2^64-2^32+1
  //
  // x3 * 2^96 + x2 * 2^64 + x1 * 2^32 + x0 \equiv
  // (x1+x2) * 2^32 + x0 - x3 - x2 mod p
  //
  // Here: x3 = 0, x2 = 0, x1 = (a >> 32), x0 = a-(x1 << 32)
  // const uint64_t p = 0xffffffff00000001;
  // uint64_t x3 = 0;
  // uint64_t x2 = 0;

  uint64_t x1 = (a >> 32);
  uint64_t x0 = (a & UINT32_MAX);

  // uint64_t res = ((x1+x2)<<32 + x0-x3-x2);
  uint64_t res = ((x1<<32) + x0);

  if(res >= PRIMEP){
    res -= PRIMEP;
    x1 = (res >> 32);
    x0 = (res & UINT32_MAX);
    res = ((x1<<32) + x0);
  }

  return res;
}

__host__ __device__  inline uint64_t s_mul(uint64_t a,uint64_t b){
  // Multiply and reduce a and b by prime 2^64-2^32+1
  #ifdef __CUDA_ARCH__
  const uint64_t GAP = (UINT64_MAX-PRIMEP+1);

  const uint64_t cHi = __umul64hi(a,b);
  const uint64_t cLo = a*b;


  // Reduce
  const uint64_t x3 = (cHi >> 32);
  const uint64_t x2 = (cHi & UINT32_MAX);
  const uint64_t x1 = (cLo >> 32);
  const uint64_t x0 = (cLo & UINT32_MAX);

  const uint64_t X1 = (x1<<32);
  const uint64_t X2 = (x2<<32);

  ///////////////////////////////
  //
  // Here we can see three kinds of overflows:
  //
  // * Negative overflow: Result is negative. 
  // Since uint64_t uses mod UINT64_MAX, we need to translate to the correct value mod PRIMEP.
  // * Simple overflow: Result is bigger than PRIMEP but not enough to exceed UINT64_MAX.
  //  We solve this in the same way we solve negative overflow, just translate to the correct value mod PRIMEP.
  // * Double overflow

  uint64_t res = X1+X2+x0-x2-x3;
  const bool testA = (x2+x3 > X1+X2+x0) && !( overflow(X1,X2) ||  overflow(X1+X2,x0) ); // Negative overflow
  const bool testB = ( res >= PRIMEP ); // Simple overflow
  const bool testC = (overflow(X1,X2) || overflow(X1+X2,x0)) && (X1+X2+x0 > x2+x3); // Double overflow

  // This avoids conditional branchs
  // res = (PRIMEP-res)*(testA) + (res-PRIMEP)*(!testA && testB) + (PRIMEP - (UINT64_MAX-res))*(!testA && !testB && testC) + (res)*(!testA && !testB && !testC);
  res =   (PRIMEP-res)*(testA) 
        + (res-PRIMEP)*(!testA && testB) 
        + (res+GAP)*(!testA && !testB && testC) 
        + (res)*(!testA && !testB && !testC);

   #else
  uint64_t res = (((__uint128_t)a) * ((__uint128_t)b) )%PRIMEP;
  #endif
  return res;
}
__host__ __device__ inline  uint64_t s_add(uint64_t a,uint64_t b){
  // Add and reduce a and b by prime 2^64-2^32+1
  // 4294967295L == UINT64_MAX - P
  uint64_t res = a+b;
  res += (res < a)*4294967295L;

  return s_rem(res);
}

__host__ __device__ inline uint64_t s_sub(uint64_t a,uint64_t b){
  // Computes a-b % P
  // 4294967295L == UINT64_MAX - P

  uint64_t res;
  // if(b > a){
  //   res = PRIMEP;
  //   res -= b;
  //   res += a;
  // }
  // else
  //   res = a-b;
  res = (a-b) + (b > a)*PRIMEP; 

  return res;
}

template <int RADIX, int type>
__device__ void butterfly(uint64_t *v){
  printf("Nothing to do");
}

template<>
__device__ inline void butterfly<2,FORWARD>(uint64_t *v){
  ///////////////////////
  // Radix-2 Butterfly //
  ///////////////////////
  const uint64_t v0 = s_rem(v[0]);
  const uint64_t v1 = s_rem(v[1]);
  v[0] = s_add(v0,v1);
  v[1] = s_sub(v0,v1);
}

template<>
__device__ inline void butterfly<2,INVERSE>(uint64_t *v){
  ///////////////////////
  // Radix-2 Butterfly //
  ///////////////////////
  const uint64_t v0 = s_rem(v[0]);
  const uint64_t v1 = s_rem(v[1]);
  v[0] = s_add(v0,v1);
  v[1] = s_sub(v0,v1);
}

template<>
__device__ inline void butterfly<4,FORWARD>(uint64_t *v){
  ///////////////////////
  // Radix-4 Butterfly //
  ///////////////////////
  // const uint64_t v0 = (v[0]);
  // const uint64_t v1 = (v[1]);
  // const uint64_t v2 = (v[2]);
  // const uint64_t v3 = (v[3]);
  
  // // v0 + v1 + v2 + v3
  // v[0] = s_add(s_add(s_add(v0,v1),v2),v3);
  // // v0 + W4*v1 - v2 - W4*v3
  // v[1] = s_sub(s_sub(s_add(v0,s_mul(W4,v1)),v2),s_mul(W4,v3)); 
  // // v0 - v1 + v2 - v3
  // v[2] = s_sub(s_add(s_sub(v0,v1),v2),v3);
  // // v0 - W4*v1 - v2 + W4*v3
  // v[3] = s_add(s_sub(s_sub(v0,s_mul(W4,v1)),v2),s_mul(W4,v3)); 
   
  register uint64_t s[4], temp;
  s[0] = s_add(v[0], v[2]);
  s[1] = s_sub(v[0], v[2]);
  s[2] = s_add(v[1], v[3]);
  s[3] = s_sub(v[1], v[3]);
  temp = s_mul(s[3], 48);
  v[0] = s_add(s[0], s[2]);
  v[1] = s_add(s[1], temp);
  v[2] = s_sub(s[0], s[2]);
  v[3] = s_sub(s[1], temp);
  
}

template<>
__device__ inline void butterfly<4,INVERSE>(uint64_t *v){
  ///////////////////////
  // Radix-4 Butterfly //
  ///////////////////////
  // const uint64_t v0 = (v[0]);
  // const uint64_t v1 = (v[1]);
  // const uint64_t v2 = (v[2]);
  // const uint64_t v3 = (v[3]);
  
  // // v0 + v1 + v2 + v3
  // v[0] = s_add(s_add(s_add(v0,v1),v2),v3);
  // // v0 - W4*v1 - v2 + W4*v3
  // v[1] = s_add(s_sub(s_sub(v0,s_mul(W4,v1)),v2),s_mul(W4,v3)); 
  // // v0 - v1 + v2 - v3
  // v[2] = s_sub(s_add(s_sub(v0,v1),v2),v3);
  // // v0 + W4*v1 - v2 - W4*v3
  // v[3] = s_sub(s_sub(s_add(v0,s_mul(W4,v1)),v2),s_mul(W4,v3));  

  register uint64_t s[4], temp;
  s[0] = s_add(v[0], v[2]);
  s[1] = s_sub(v[0], v[2]);
  s[2] = s_add(v[1], v[3]);
  s[3] = s_sub(v[1], v[3]);
  temp = s_mul(s[3], W4);
  v[0] = s_add(s[0], s[2]);
  v[1] = s_sub(s[1], temp);
  v[2] = s_sub(s[0], s[2]);
  v[3] = s_add(s[1], temp);
}

template<>
__device__ void butterfly<8,FORWARD>(uint64_t *v){
  ///////////////////////
  // Radix-4 Butterfly //
  ///////////////////////
  
  // register uint64 s[8], temp;
  // s[0] = s_add(v[0], v[4]);
  // s[1] = s_sub(v[0], v[4]);
  // s[2] = s_add(v[2], v[6]);
  // s[3] = s_sub(v[2], v[6]);
  // s[4] = s_add(v[1], v[5]);
  // s[5] = s_sub(v[1], v[5]);
  // s[6] = s_add(v[3], v[7]);
  // s[7] = s_sub(v[3], v[7]);
  // v[0] = s_add(s[0], s[2]);
  // v[2] = s_sub(s[0], s[2]);
  // temp = s_mul(s[3], W4);
  // v[1] = s_add(s[1], temp);
  // v[3] = s_sub(s[1], temp);
  // v[4] = s_add(s[4], s[6]);
  // v[6] = s_sub(s[4], s[6]);
  // temp = s_mul(s[7], W4);
  // v[5] = s_add(s[5], temp);
  // v[7] = s_sub(s[5], temp);
  // s[0] = s_add(v[0], v[4]);
  // s[4] = s_sub(v[0], v[4]);
  // temp = s_mul(v[5], W2);
  // s[1] = s_add(v[1], temp);
  // s[5] = s_sub(v[1], temp);
  // temp = s_mul(v[6], W4);
  // s[2] = s_add(v[2], temp);
  // s[6] = s_sub(v[2], temp);
  // temp = s_mul(v[7], 72);
  // s[3] = s_add(v[3], temp);
  // s[7] = s_sub(v[3], temp);
  // v[0] = s[0];
  // v[1] = s[1];
  // v[2] = s[2];
  // v[3] = s[3];
  // v[4] = s[4];
  // v[5] = s[5];
  // v[6] = s[6];
  // v[7] = s[7];
}

// template<>
// __device__ void butterfly<8,INVERSE>(uint64_t *v){
//   ///////////////////////
//   // Radix-4 Butterfly //
//   ///////////////////////
//   const uint64_t v0 = s_rem(v[0]);
//   const uint64_t v1 = s_rem(v[1]);
//   const uint64_t v2 = s_rem(v[2]);
//   const uint64_t v3 = s_rem(v[3]);
//   const uint64_t v4 = s_rem(v[4]);
//   const uint64_t v5 = s_rem(v[5]);
//   const uint64_t v6 = s_rem(v[6]);
//   const uint64_t v7 = s_rem(v[7]);
  

//   v[0] = s_add(s_add(s_add(s_add(s_add(s_add(s_add(v0,v1),v2),v3),v4),v5),v6),v7);
//   v[1] = s_add(s_add(s_add(s_add(s_add(s_add(s_add( v0,
//                                                     s_mul(v1,WInv8[1])),
//                                                     s_mul(v2,WInv8[2])),
//                                                     s_mul(v3,WInv8[3])),
//                                                     s_mul(v4,WInv8[4])),
//                                                     s_mul(v5,WInv8[5])),
//                                                     s_mul(v6,WInv8[6])),
//                                                     s_mul(v7,WInv8[7]));
//   v[2] = s_add(s_add(s_add(s_add(s_add(s_add(s_add( v0,
//                                                     s_mul(v1,WInv8[2])),
//                                                     s_mul(v2,WInv8[4])),
//                                                     s_mul(v3,WInv8[6])),
//                                                     s_mul(v4,WInv8[8])),
//                                                     s_mul(v5,WInv8[10])),
//                                                     s_mul(v6,WInv8[12])),
//                                                     s_mul(v7,WInv8[14]));
//   v[3] = s_add(s_add(s_add(s_add(s_add(s_add(s_add( v0,
//                                                     s_mul(v1,WInv8[3])),
//                                                     s_mul(v2,WInv8[6])),
//                                                     s_mul(v3,WInv8[9])),
//                                                     s_mul(v4,WInv8[12])),
//                                                     s_mul(v5,WInv8[15])),
//                                                     s_mul(v6,WInv8[18])),
//                                                     s_mul(v7,WInv8[21]));
//   v[4] = s_add(s_add(s_add(s_add(s_add(s_add(s_add( v0,
//                                                     s_mul(v1,WInv8[4])),
//                                                     s_mul(v2,WInv8[8])),
//                                                     s_mul(v3,WInv8[12])),
//                                                     s_mul(v4,WInv8[16])),
//                                                     s_mul(v5,WInv8[20])),
//                                                     s_mul(v6,WInv8[24])),
//                                                     s_mul(v7,WInv8[28]));
//   v[5] = s_add(s_add(s_add(s_add(s_add(s_add(s_add( v0,
//                                                     s_mul(v1,WInv8[5])),
//                                                     s_mul(v2,WInv8[10])),
//                                                     s_mul(v3,WInv8[15])),
//                                                     s_mul(v4,WInv8[20])),
//                                                     s_mul(v5,WInv8[25])),
//                                                     s_mul(v6,WInv8[30])),
//                                                     s_mul(v7,WInv8[35]));
//   v[6] = s_add(s_add(s_add(s_add(s_add(s_add(s_add( v0,
//                                                     s_mul(v1,WInv8[6])),
//                                                     s_mul(v2,WInv8[12])),
//                                                     s_mul(v3,WInv8[18])),
//                                                     s_mul(v4,WInv8[24])),
//                                                     s_mul(v5,WInv8[30])),
//                                                     s_mul(v6,WInv8[36])),
//                                                     s_mul(v7,WInv8[42]));
//   v[7] = s_add(s_add(s_add(s_add(s_add(s_add(s_add( v0,
//                                                     s_mul(v1,WInv8[7])),
//                                                     s_mul(v2,WInv8[14])),
//                                                     s_mul(v3,WInv8[21])),
//                                                     s_mul(v4,WInv8[28])),
//                                                     s_mul(v5,WInv8[35])),
//                                                     s_mul(v6,WInv8[42])),
//                                                     s_mul(v7,WInv8[49]));
// }

__host__ __device__ int expand(int idxL, int N1, int N2){
	return (idxL/N1)*N1*N2 + (idxL%N1);
}

__global__ void NTTScale(cuyasheint_t *data,const int size,const int N){
  const unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
  // const unsigned int cid = (tid/N)*N + (tid%N); // residueId*resideSize + coefficient 
  if( tid < size )
    data[tid] /= N;
} 

template<int RADIX, int type>
__host__ __device__ void NTTIteration(cuyasheint_t *W,
                                      cuyasheint_t *WInv,
                                      const int residue_index,
                                      const int j,
                                      const int N,
                                      const int Ns,
                                      const cuyasheint_t* data0,
                                      cuyasheint_t *data1){
	uint64_t v[RADIX] = {0};
	const int idxS = j+residue_index;
  int w_index = ((j%Ns)*N)/(Ns*RADIX);
  const int idxD = expand(j,Ns,RADIX)+residue_index;
  if(idxD == 151 || idxD+Ns == 151)
    v[0] = v[0];

  for(int r=0; r<RADIX; r++)
    if(type == FORWARD)
      v[r] = s_mul(data0[idxS+r*N/RADIX],W[w_index*r]);
    else
      v[r] = s_mul(data0[idxS+r*N/RADIX],WInv[w_index*r]);

  butterfly<RADIX,type>(v);

	// const int idxD = expand(j,Ns,RADIX)+residue_index;
	for(int r=0; r<RADIX;r++)
  		data1[idxD+r*Ns] = v[r];
  
}

template<int RADIX, int type>
__global__ void NTT(cuyasheint_t *d_W,cuyasheint_t *d_WInv,const int N, const int Ns, cuyasheint_t* dataI, cuyasheint_t* dataO){

  const int residue_index = (blockIdx.x)*N;
  for(int i = 0; i < N/RADIX; i += 1024){
    // " Threads virtuais "
    const int j = (threadIdx.x+i);
    if( j < N)
      NTTIteration<RADIX,type>(d_W,d_WInv,residue_index,j, N, Ns, dataI, dataO);
    __syncthreads();
  }
}

__host__ void CUDAFunctions::callNTT(const int N, const int NPolis,int RADIX, cuyasheint_t* dataI, cuyasheint_t* dataO,const int type){

  dim3 blockDim(std::min(N/RADIX,1024));
  dim3 gridDim(NPolis);

  for(int Ns=1; Ns<N; Ns*=RADIX){
    if(RADIX == 4){
      if(type == FORWARD)
        NTT<4,FORWARD><<<gridDim,blockDim>>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,Ns,dataI,dataO);
      else
        NTT<4,INVERSE><<<gridDim,blockDim>>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,Ns,dataI,dataO);
    }
    else{
      assert(RADIX == 2);
      if(type == FORWARD)
        NTT<2,FORWARD><<<gridDim,blockDim>>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,Ns,dataI,dataO);
      else
        NTT<2,INVERSE><<<gridDim,blockDim>>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,Ns,dataI,dataO);
    }
    assert(cudaGetLastError() == cudaSuccess);
    std::swap(dataI,dataO);
  }
}

__global__ void polynomialNTTMul(cuyasheint_t *c, const cuyasheint_t *a,const cuyasheint_t *b,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      uint64_t a_value = a[tid];
      uint64_t b_value = b[tid];

      c[tid] = s_mul(a_value,b_value);
  }
}

__global__ void polynomialNTTAdd(cuyasheint_t *a,const cuyasheint_t *b,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      uint64_t a_value = a[tid];
      uint64_t b_value = b[tid];

      // In-place
      a[tid] = s_add(a_value,b_value);
      // a[tid] = a_value*b_value % 18446744069414584321;
  }
}
// #endif

__global__ void polynomialOPInteger(const int opcode,
                                      const cuyasheint_t *a,
                                      const cuyasheint_t integer_array,
                                      cuyasheint_t *output,
                                      const int N,
                                      const int NPolis){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int size = N*NPolis;
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int cid = tid % N; // Coefficient id
  const int rid = tid / N; // Residue id

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
    cuyasheint_t operand = integer_array;

    switch(opcode)
    {
    case ADD:
      if(cid == 0)
        output[tid] = (a[tid] + operand) % CRTPrimesConstant[rid];
      break;
    case SUB:
      if(cid == 0){
        if(a[tid] > operand)
          output[tid] = (a[tid] - operand) % CRTPrimesConstant[rid];
        else
          output[tid] = ((CRTPrimesConstant[rid]-operand) + a[tid]) % CRTPrimesConstant[rid];
      }
      break;
    case MUL:
        output[tid] = (a[tid] * operand)% CRTPrimesConstant[rid];
      break;
    default:
      //This case shouldn't be used. 
      assert(1 == 0);
      break;
    }
  }

}

__host__ cuyasheint_t* CUDAFunctions::callPolynomialOPInteger(
                                                              const int opcode,
                                                              cudaStream_t stream,
                                                              cuyasheint_t *a,
                                                              cuyasheint_t integer_array,
                                                              const int N,
                                                              const int NPolis)
{
  // This method applies a 0-degree operation over all CRT residues
  const int size = N*NPolis;

  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  cuyasheint_t *d_pointer;
  cudaError_t result = cudaMalloc((void**)&d_pointer,
                                  size*sizeof(cuyasheint_t));        
  assert(result == cudaSuccess);

  polynomialOPInteger<<< gridDim,blockDim, 0, stream>>> ( opcode,
                                                          a,
                                                          integer_array,
                                                          d_pointer,
                                                          N,
                                                          NPolis);
  assert(cudaGetLastError() == cudaSuccess);

  return d_pointer;
}
__host__ void CUDAFunctions::callPolynomialOPIntegerInplace(
                                                              const int opcode,
                                                              cudaStream_t stream,
                                                              cuyasheint_t *a,
                                                              cuyasheint_t integer_array,
                                                              const int N,
                                                              const int NPolis)
{
  // This method applies a 0-degree operation over all CRT residues
  const int size = N*NPolis;

  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomialOPInteger<<< gridDim,blockDim, 0, stream>>> ( opcode,
                                                          a,
                                                          integer_array,
                                                          a,
                                                          N,
                                                          NPolis);
  assert(cudaGetLastError() == cudaSuccess);
  return;
}

__global__ void polynomialcuFFTOPInteger( const int opcode,
                                          const Complex *a,
                                          const cuyasheint_t integer,
                                          Complex *output,
                                          const int N,
                                          const int NPolis){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int size = N*NPolis;
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int cid = tid % N; // Coefficient id
  // const int rid = tid / N; // Residue id

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
    Complex operand = {__ull2double_rn(integer),0};

    switch(opcode)
    {
    case ADD:
      if(cid == 0)
        output[tid] = ComplexAdd(a[tid],operand) ;
      break;
    case SUB:
      if(cid == 0){
        output[tid] = ComplexSub(a[tid],operand);
      }
      break;
    case MUL:
        output[tid] = ComplexMul(a[tid] , operand);
      break;
    default:
      //This case shouldn't be used. 
      assert(1 == 0);
      break;
    }
  }

}

__host__ Complex* CUDAFunctions::callPolynomialcuFFTOPInteger(
                                                              const int opcode,
                                                              cudaStream_t stream,
                                                              Complex *a,
                                                              cuyasheint_t integer,
                                                              const int N,
                                                              const int NPolis)
{
  // This method applies a 0-degree operation over all CRT residues
  const int size = N*NPolis;

  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  Complex *d_pointer;
  cudaError_t result = cudaMalloc((void**)&d_pointer,
                                  size*sizeof(Complex));        
  assert(result == cudaSuccess);

  polynomialcuFFTOPInteger<<< gridDim,blockDim, 0, stream>>> ( opcode,
                                                          a,
                                                          integer,
                                                          d_pointer,
                                                          N,
                                                          NPolis);
  assert(cudaGetLastError() == cudaSuccess);

  return d_pointer;
}

__host__ void CUDAFunctions::callPolynomialcuFFTOPIntegerInplace(
                                                                      const int opcode,
                                                                      cudaStream_t stream,
                                                                      Complex *a,
                                                                      cuyasheint_t integer,
                                                                      const int N,
                                                                      const int NPolis)
{
  // This method applies a 0-degree operation over all CRT residues
  const int size = N*NPolis;

  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomialcuFFTOPInteger<<< gridDim,blockDim, 0, stream>>> ( opcode,
                                                                a,
                                                                integer,
                                                                a,
                                                                N,
                                                                NPolis);
  assert(cudaGetLastError() == cudaSuccess);

  return;
}

__global__ void polynomialOPDigit(const int opcode,
                                      bn_t *a,
                                      const bn_t digit,
                                      const int N
                                  ){

  // We have one thread per polynomial coefficient on 32 threads-block.
  const int size = N;
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  int nwords = 0;
  cuyasheint_t carry = 0;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
    switch(opcode)
    {
    case ADD:
      if(tid == 0){

        nwords = max_d(a[tid].used,digit.used);
        carry = bn_addn_low(a[tid].dp, a[tid].dp, digit.dp,nwords);
        a[tid].used = nwords;

        /* Equivalent to "If has a carry, add as last word" */
        a[tid].dp[a[tid].used] = carry;
        a[tid].used += (carry > 0);
      }
      break;
    case MUL:
      assert(a[tid].alloc >= STD_BNT_WORDS_ALLOC);
      assert(digit.alloc >= STD_BNT_WORDS_ALLOC);

      bn_muln_low(a[tid].dp,
                  a[tid].dp,
                  digit.dp,
                  STD_BNT_WORDS_ALLOC);
      break;
    default:
      //This case shouldn't be used. 
      assert(1 == 0);
      break;
    }
  }

}

__host__ void CUDAFunctions::callPolynomialOPDigit( const int opcode,
                                                            cudaStream_t stream,
                                                            bn_t *a,
                                                            bn_t digit,
                                                            const int N){
    // This method applies a 0-degree operation over all coeficients
  const int size = N;

  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomialOPDigit<<< gridDim,blockDim, 1, stream>>> ( opcode,
                                                        a,
                                                        digit,
                                                        N);
  assert(cudaGetLastError() == cudaSuccess);
  return;
}

__host__ cuyasheint_t* CUDAFunctions::applyNTT( cuyasheint_t *d_a,
                                                const int N,
                                                const int NPolis,
                                                int type,
                                                cudaStream_t stream){
  if(N != CUDAFunctions::N)
    CUDAFunctions::init(N/2);

  cudaError_t result;
  const int size = N*NPolis;
  // cuyasheint_t *aux = CUDAFunctions::d_mulAux;
  cuyasheint_t *aux;
  result = cudaMalloc((void**)&aux,size*sizeof(cuyasheint_t));

  result = cudaMemsetAsync(aux,0,size*sizeof(cuyasheint_t),stream);
  assert(result == cudaSuccess);

  int RADIX;
  /*if(N % 8 == 0)
    RADIX = 8;
  else*/
  if(is_power_of(N,4))
    RADIX = 4;
  else{
    assert(is_power_of(N,2));
    RADIX = 2;
    }
  dim3 blockDim(std::min(N/RADIX,1024));
  dim3 gridDim(NPolis);

  // Forward
  for(int Ns=1; Ns<N; Ns*=RADIX){
    if(type == FORWARD){ 
      /*if(RADIX == 8)
        NTT<8,FORWARD><<<gridDim,blockDim,1,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,Ns,d_a,aux);
      else*/ if(RADIX == 4)
        NTT<4,FORWARD><<<gridDim,blockDim,0,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,Ns,d_a,aux);
      else{
        assert(RADIX == 2);
        NTT<2,FORWARD><<<gridDim,blockDim,0,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,Ns,d_a,aux);
      }
    }else{      
      /*if(RADIX == 8)
        NTT<8,FORWARD><<<gridDim,blockDim,0,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,Ns,d_a,aux);
      else*/ if(RADIX == 4)
        NTT<4,INVERSE><<<gridDim,blockDim,0,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,Ns,d_a,aux);
      else{
        assert(RADIX == 2);
        NTT<2,INVERSE><<<gridDim,blockDim,0,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,Ns,d_a,aux);
      }
    }
    assert(cudaGetLastError() == cudaSuccess);
    std::swap(aux,d_a);
  }
  if(type == INVERSE){
    // std::swap(aux,d_a);
    dim3 blockDimMul(ADDBLOCKXDIM);
    dim3 gridDimMul((size)/ADDBLOCKXDIM+1); // We expect that ADDBLOCKXDIM always divide size
    NTTScale<<< gridDimMul,blockDimMul,0,stream >>>(d_a,size,N);
    assert(cudaGetLastError() == cudaSuccess);
  }
  return d_a;
}

__host__ void CUDAFunctions::executeCopyIntegerToComplex(   Complex *d_a, 
                                                            cuyasheint_t *a,
                                                            const int size,
                                                            cudaStream_t stream){
  dim3 blockDim(32);
  dim3 gridDim(size/32 + (size % 32 == 0? 0:1));

  copyIntegerToComplex<<< gridDim,blockDim,0,stream >>>(d_a,a,size);

  assert(cudaGetLastError() == cudaSuccess);
}

__host__ void CUDAFunctions::executeCopyAndNormalizeComplexRealPartToInteger(   cuyasheint_t *d_a, 
                                                                                cufftDoubleComplex *a,
                                                                                const int size,
                                                                                int N,
                                                                                cudaStream_t stream){
  dim3 blockDim(32);
  dim3 gridDim(size/32 + (size % 32 == 0? 0:1));

  copyAndNormalizeComplexRealPartToInteger<<< gridDim,blockDim,0,stream >>>( d_a,
                                                                            a,
                                                                            size,
                                                                            N);

  assert(cudaGetLastError() == cudaSuccess);
}

__host__ void CUDAFunctions::executeNTTScale(   cuyasheint_t *d_result, 
                                                const int size, 
                                                const int N,
                                                cudaStream_t stream){
  dim3 blockDimMul(ADDBLOCKXDIM);
  dim3 gridDimMul((size)/ADDBLOCKXDIM+1); // We expect that ADDBLOCKXDIM always divide size
  NTTScale<<< gridDimMul,blockDimMul,0,stream >>>(d_result,size,N);
  assert(cudaGetLastError() == cudaSuccess);
}

__host__ void CUDAFunctions::executeCuFFTPolynomialMul( Complex *c, 
                                                        Complex *a, 
                                                        Complex *b, 
                                                        int size, 
                                                        cudaStream_t stream){
  dim3 blockDim(32);
  dim3 gridDim(size/32 + (size % 32 == 0? 0:1));

  polynomialcuFFTMul<<<gridDim,blockDim,0,stream>>>(c,a,b,size);

  assert(cudaGetLastError() == cudaSuccess);
}
__host__ void CUDAFunctions::executePolynomialMul(cuyasheint_t *c, 
                                                  cuyasheint_t *a, 
                                                  cuyasheint_t *b, 
                                                  const int size, 
                                                  cudaStream_t stream){
  dim3 blockDimMul(ADDBLOCKXDIM);
  dim3 gridDimMul((size)/ADDBLOCKXDIM+1); // We expect that ADDBLOCKXDIM always divide size
  polynomialNTTMul<<<gridDimMul,blockDimMul,0,stream>>>(c,a,b,size);
  assert(cudaGetLastError() == cudaSuccess);
}

__host__ void CUDAFunctions::executePolynomialAdd(cuyasheint_t *c, 
                                                  cuyasheint_t *a, 
                                                  cuyasheint_t *b, 
                                                  const int size, 
                                                  cudaStream_t stream){
  dim3 blockDimMul(ADDBLOCKXDIM);
  dim3 gridDimMul((size)/ADDBLOCKXDIM+1); // We expect that ADDBLOCKXDIM always divide size
  polynomialNTTAdd<<<gridDimMul,blockDimMul,0,stream>>>(a,b,size);
  assert(cudaGetLastError() == cudaSuccess);
}

/**
 * Returns true if a is power of b
 * @param  a [description]
 * @param  b [description]
 * @return   [description]
 */
__host__ bool is_power_of(uint64_t a, uint64_t b){
  assert(b > 1);
  
  uint64_t n = a;
  while (n % b == 0)
    n /= b;
  return (n==1);
}



__host__ cuyasheint_t* CUDAFunctions::callPolynomialMul(cuyasheint_t *output,
                                                        cuyasheint_t *a,
                                                        cuyasheint_t *b,
                                                        const int size,
                                                        cudaStream_t stream){
  // This method expects that both arrays are aligned

  // Input:
  // stream: cudaStream
  // a: first operand
  // realign_A: flag. true if this operand need to be realign
  // A_N: number of coefficients for each operand. Used only if we need to realign this
  // b: second operand
  // realign_B: flag. true if this operand need to be realign
  // B_N: number of coefficients for each residue. Used only if we need to realign this
  // N: number of coefficients for each residue. This is the N that should be considered for the operation.
  // NPolis: number of residues
  // All representations should be concatenated aligned
  assert((N>0)&&((N & (N - 1)) == 0));//Check if N is power of 2
  assert(N == CUDAFunctions::N);
  cuyasheint_t *d_result = output;
  // cudaError_t result;

  // #ifdef NTTMUL
  if(transform == NTTMUL){

    // Multiply
    dim3 blockDimMul(ADDBLOCKXDIM);
    dim3 gridDimMul((size)/ADDBLOCKXDIM+1); // We expect that ADDBLOCKXDIM always divide size
    polynomialNTTMul<<<gridDimMul,blockDimMul,0,stream>>>(output,a,b,size);
    assert(cudaGetLastError() == cudaSuccess);

    // result = cudaDeviceSynchronize();
  }else{
  
    /////////////
    // cuFFT  //
    ///////////
    // dim3 blockDim(32);
    // dim3 gridDim(size/32 + (size % 32 == 0? 0:1));

    // polynomialcuFFTMul<<<gridDim,blockDim,0,stream>>>(d_a,d_b,d_c,size);
    // assert(cudaGetLastError() == cudaSuccess);
  }

  return d_result;
}

/**
 * [CUDAFunctions::init description]
 * @param N The target polynomial degree
 */
__host__ void CUDAFunctions::init(int M){
  int N = 2*M;
  CUDAFunctions::N = N;
  cudaError_t result;

  // #ifdef NTTMUL
  // W used on NTT
  std::cout << "Will compute W -- N = " << N << std::endl;
  #ifdef VERBOSE
  std::cout << "P " << PZZ << std::endl;
  #endif

  cuyasheint_t *h_W;
  cuyasheint_t *h_WInv;

  assert((PZZ-1)%(N) == 0);

  cuyasheint_t k = conv<cuyasheint_t>(PZZ-1)/N;
  ZZ wNZZ = NTL::PowerMod(ZZ(PRIMITIVE_ROOT),k,PZZ);

  wN = conv<cuyasheint_t>(wNZZ);
  h_W = (cuyasheint_t*)malloc(N*sizeof(cuyasheint_t));
  result = cudaMalloc((void**)&d_W,N*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);
  h_WInv = (cuyasheint_t*)malloc(N*sizeof(cuyasheint_t));
  result = cudaMalloc((void**)&d_WInv,N*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);

  // Computes 1-th column from W
  for(int j = 0; j < N; j++)
    h_W[j] = conv<cuyasheint_t>(NTL::PowerMod(wNZZ,j,PZZ));
  

  // Computes 1-th column from WInv
  for(int j = 0; j < N; j++)
      h_WInv[j] = conv<cuyasheint_t>(NTL::InvMod(conv<ZZ>(h_W[j]),PZZ ));

  result = cudaMemcpy (d_W,h_W , N*sizeof(cuyasheint_t),cudaMemcpyHostToDevice);
  assert(result == cudaSuccess);
  result = cudaMemcpy(d_WInv,h_WInv , N*sizeof(cuyasheint_t),cudaMemcpyHostToDevice);
  assert(result == cudaSuccess);

  free(h_W);
  free(h_WInv);

  // int RADIX_N = 8;
  // // W used on radix-8 NTT
  // #ifdef VERBOSE
  // std::cout << "Will compute W -- N = 8" << std::endl;
  // std::cout << "P " << PZZ << std::endl;
  // #endif

  // k = conv<cuyasheint_t>(PZZ-1)/RADIX_N;
  // wNZZ = NTL::PowerMod(ZZ(PRIMITIVE_ROOT),k,PZZ);

  // wN = conv<cuyasheint_t>(wNZZ);
  
  // int VALUES_TO_COMPUTE = (RADIX_N - 1)*(RADIX_N - 1)+1;
  // h_W = (cuyasheint_t*)malloc(VALUES_TO_COMPUTE*sizeof(cuyasheint_t));
  // h_WInv = (cuyasheint_t*)malloc(VALUES_TO_COMPUTE*sizeof(cuyasheint_t));

  // // Computes 1-th column from W
  // for(int j = 0; j < VALUES_TO_COMPUTE; j++)
  //   h_W[j] = conv<cuyasheint_t>(NTL::PowerMod(wNZZ,j,PZZ));

  // // Computes 1-th column from WInv
  // for(int j = 0; j < VALUES_TO_COMPUTE; j++)
  //     h_WInv[j] = conv<cuyasheint_t>(NTL::InvMod(conv<ZZ>(h_W[j]),PZZ ));

  // result = cudaMemcpyToSymbol (W8,h_W, VALUES_TO_COMPUTE*sizeof(cuyasheint_t));
  // assert(result == cudaSuccess);
  // result = cudaMemcpyToSymbol (WInv8,h_WInv, VALUES_TO_COMPUTE*sizeof(cuyasheint_t));
  // assert(result == cudaSuccess);

  // free(h_W);
  // free(h_WInv);

  // RADIX_N = 16;
  // // W used on radix-16 NTT
  // #ifdef VERBOSE
  // std::cout << "Will compute W -- N = 16" << std::endl;
  // std::cout << "P " << PZZ << std::endl;
  // #endif

  // k = conv<cuyasheint_t>(PZZ-1)/RADIX_N;
  // wNZZ = NTL::PowerMod(ZZ(PRIMITIVE_ROOT),k,PZZ);

  // wN = conv<cuyasheint_t>(wNZZ);
  
  // VALUES_TO_COMPUTE = (RADIX_N - 1)*(RADIX_N - 1)+1;
  // h_W = (cuyasheint_t*)malloc(VALUES_TO_COMPUTE*sizeof(cuyasheint_t));
  // h_WInv = (cuyasheint_t*)malloc(VALUES_TO_COMPUTE*sizeof(cuyasheint_t));

  // // Computes 1-th column from W
  // for(int j = 0; j < VALUES_TO_COMPUTE; j++)
  //   h_W[j] = conv<cuyasheint_t>(NTL::PowerMod(wNZZ,j,PZZ));

  // // Computes 1-th column from WInv
  // for(int j = 0; j < VALUES_TO_COMPUTE; j++)
  //     h_WInv[j] = conv<cuyasheint_t>(NTL::InvMod(conv<ZZ>(h_W[j]),PZZ ));

  // result = cudaMemcpyToSymbol (W8,h_W, VALUES_TO_COMPUTE*sizeof(cuyasheint_t));
  // assert(result == cudaSuccess);
  // result = cudaMemcpyToSymbol (WInv8,h_WInv, VALUES_TO_COMPUTE*sizeof(cuyasheint_t));
  // assert(result == cudaSuccess);

  // free(h_W);
  // free(h_WInv);

  cufftResult fftResult;

    // # of CRT residues
  const int batch = Polynomial::CRTPrimes.size();
  assert(batch > 0);

  // # 1 dimensional FFT
  const int rank = 1;

  // No idea what is this
  int n[1] = {N};


  fftResult = cufftPlanMany(&CUDAFunctions::plan, rank, n,
       NULL, 1, N,  //advanced data layout, NULL shuts it off
       NULL, 1, N,  //advanced data layout, NULL shuts it off
       CUFFT_Z2Z, batch);
  // fftResult = cufftPlan1d(&CUDAFunctions::plan, N, CUFFT_Z2Z, 1);


  assert(fftResult == CUFFT_SUCCESS);
  std::cout << "Plan created with signal size " << N << std::endl;
  /**
   * Alloc memory for d_inner_results
   */
  const unsigned int size = N*Polynomial::CRTPrimes.size();

  result = cudaMalloc((void**)&CUDAFunctions::d_inner_results, size*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);
  result = cudaMalloc((void**)&CUDAFunctions::d_inner_results_used, size*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);


    /**
     * Pre-allocated arrays for NTT multiplication
     */
    
  result = cudaMalloc((void**)&CUDAFunctions::d_mulA,size*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);
  result = cudaMalloc((void**)&CUDAFunctions::d_mulB,size*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);
  result = cudaMalloc((void**)&CUDAFunctions::d_mulAux,size*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);


    /**
     * Pre-allocated arrays for FFT multiplication
     */
    
  result = cudaMalloc((void**)&CUDAFunctions::d_mulComplexA,size*sizeof(Complex));
  assert(result == cudaSuccess);
  result = cudaMalloc((void**)&CUDAFunctions::d_mulComplexB,size*sizeof(Complex));
  assert(result == cudaSuccess);
  result = cudaMalloc((void**)&CUDAFunctions::d_mulComplexC,size*sizeof(Complex));
  assert(result == cudaSuccess);
}

__global__ void cuICRTFix(bn_t *a, const int N, bn_t q,bn_t u_q,bn_t q2){
  //////////////////////////////////////////////////////
  // This kernel must be executed with N threads //
  //////////////////////////////////////////////////////
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if(tid < N){
    bn_t coef = a[tid];
    coef.used = get_used_index(coef.dp,STD_BNT_WORDS_ALLOC)+1;

    if(bn_cmp_abs(&coef,&q2) == CMP_GT){
      /**
       * If coef > q^2, it is result of an underflow on polynomial reduction 
       * during the polynomial reduction.
       *
       * coef = q - (M - coef)
       */
      // result = M - coef
      int carry = bn_subn_low(coef.dp, M, coef.dp, max_d(coef.used,M_used));
      assert(carry == BN_POS);     
      coef.used = get_used_index(coef.dp,STD_BNT_WORDS_ALLOC)+1;
      // result = q - result
      carry = bn_subn_low(coef.dp,q.dp,coef.dp,max_d(coef.used,q.used));
      coef.used = get_used_index(coef.dp,STD_BNT_WORDS_ALLOC)+1;
    }  
    a[tid] = coef;
    // result = result % q
    bn_rem( a,
                  a,
                  N,
                  q.dp,
                  q.used,
                  u_q.dp,
                  u_q.used); 
    bn_zero_non_used(&a[tid]);
  }
}

__global__ void polynomialReductionCRT(cuyasheint_t *a,const int half,const int N,const int NPolis){     
  // This kernel must have (N-half)*Npolis threads

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int rid = tid / (N-half); 
  const int cid = tid % (N-half);

  if( (cid+half+1 < N) && (rid*N + cid + half + 1 < N*NPolis)){
    // assert(a[rid*N + cid] < CRTPrimesConstant[rid]);
    // assert(a[rid*N + cid + half + 1] < CRTPrimesConstant[rid]);
    a[rid*N + cid] %= CRTPrimesConstant[rid];
    a[rid*N + cid + half + 1] %= CRTPrimesConstant[rid];

    // bool is_neg = (a[rid*N + cid] < a[rid*N + cid + half + 1]);
    a[rid*N + cid] -= a[rid*N + cid + half + 1];
    // a[rid*N + cid] += is_neg*CRTPrimesConstant[rid]*CRTPrimesConstant[rid];
    __syncthreads();
    a[rid*N + cid + half + 1] = 0;
    a[rid*N + cid] %= CRTPrimesConstant[rid];
  }

}

__global__ void polynomialReductionCoefs(bn_t *a,const int half,const int N,const bn_t q){     
  ////////////////////////////////////////////////////////
  // This kernel must be executed with (N-half) threads //
  ////////////////////////////////////////////////////////

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int cid = tid % (N-half);

  if(cid + half + 1 < N){

    // a[i] = a[i] - a[i+half]
    int carry = bn_subn_low(a[cid].dp, a[cid].dp, a[cid + half + 1].dp, min_d(a[cid].used,a[cid + half + 1].used));
    a[cid].used = min_d(a[cid].used,a[cid + half + 1].used);
    
    if(carry == BN_NEG){
      // q - (UINT64_MAX - c)
      // compl2 == UINT64_MAX -C
      
      // SOMAR Q
      // carry = bn_addn_low(a[cid].dp,q.dp,a[cid].dp,q.used);
      // a[cid].used = max_d(a[cid].used,q.used);
      // a[cid].dp[a[cid].used] = carry;
      // a[cid].used += (carry > 0);
      // bn_zero_non_used(&a[cid]);
      
      
      bn_2_compl(&a[cid]);
      carry = bn_subn_low(a[cid].dp,q.dp,a[cid].dp,q.used);
      assert(carry == BN_POS);
      a[cid].used = q.used;
      bn_zero_non_used(&a[cid]);
    }
    

    __syncthreads();
    bn_zero(&a[cid + half + 1]);
  }
}

__host__ void Polynomial::reduce(){
  // Just like DivRem, but here we reduce with a cyclotomic polynomial
  
  //////////////////////////
  // Polynomial reduction //
  //////////////////////////

  Polynomial *phi = (Polynomial::global_phi);
  ZZ q = (Polynomial::global_mod);
  
  // Until we debug reduction on GPU, we need this
  // #warning Polynomial reduction forced to HOST
  update_host_data();
  set_crt_computed(false);
  set_icrt_computed(false);
  set_transf_computed(false);
  set_itransf_computed(false);

  if(!(this->get_crt_computed() || this->get_icrt_computed() || this->get_transf_computed())){
    #ifdef VERBOSE
    std::cout << "Reduce on host." << std::endl;
    #endif
    /**
     * Reduce on host
     */
    if(check_special_rem_format(phi)){
      #ifdef VERBOSE
      std::cout << "Rem in special mode."<<std::endl;
      #endif

      const unsigned int half = phi->deg()-1;     

      // #pragma omp parallel for
      for(unsigned int i = 0;(i <= half) && (i + half + 1 <= deg());i++){
        this->set_coeff(i,this->get_coeff(i)-this->get_coeff(i+half+1));
        this->set_coeff(i+half+1,0);
      }
    }else{
      throw "Reduce: I don't know how to compute this!";
    }
    *this %= q;
    set_crt_computed(false);
    set_icrt_computed(false);
    set_transf_computed(false);
    set_itransf_computed(false);
    set_host_updated(true);
    normalize();
    update_crt_spacing(2*phi->deg());
  }else{

    #ifdef VERBOSE
    std::cout << "Reduce on device." << std::endl;
    #endif
    /**
     * Reduce on devicce
     */
    #if PREDUCTION == RESIDUES
     //////////////////////////////////////////////////
     // Polynomial reduction applied on CRT residues //
     //////////////////////////////////////////////////
    const int half = phi->deg()-1;
    const int N = get_crt_spacing();
    const int NPolis = CRTPrimes.size();
    int size = (N-half)*NPolis;

    if(size > 0){
      itransf();

      dim3 blockDim(ADDBLOCKXDIM);
      dim3 gridDim(size/ADDBLOCKXDIM + (size % ADDBLOCKXDIM == 0? 0:1));
      /**
       * Polynomial reduction
       */
      polynomialReductionCRT<<< gridDim,blockDim, 0, get_stream()>>>( get_device_crt_residues(),
                                                                            half,
                                                                            N,
                                                                            NPolis);
      cudaError_t result = cudaGetLastError();
      assert(result == cudaSuccess);
      
      /**
       * The polynomial reduction on CRT residues may generate erroneous values because of underflows.
       * To solve this we must apply ICRT at this point and call a routine to adjust the coefficients.  
       */
      set_crt_computed(true);
      set_icrt_computed(false);
      set_transf_computed(false);
      set_itransf_computed(false);
      set_host_updated(false);
      
      icrt();
      bn_t Q;
      get_words(&Q,q);
      bn_t Q2;
      get_words(&Q2,q*q);
    
      size = N;
      dim3 gridDimFix(size/ADDBLOCKXDIM + (size % ADDBLOCKXDIM == 0? 0:1));
      cuICRTFix<<< gridDimFix,blockDim,0,get_stream()>>>(d_bn_coefs, N, Q,get_reciprocal(q),Q2);      

      set_crt_computed(false);
      set_icrt_computed(true);
      set_transf_computed(false);
      set_itransf_computed(false);
      set_host_updated(false);

    }
    #else
    //////////////////////////////////////////////////
    // Polynomial reduction applied on coefficients //
    //////////////////////////////////////////////////
    icrt();
    modn(q);

    const int half = phi->deg()-1;
    const int N = deg()+1; // Number of coefficients
    const int size = (N-half);

    if(size > 0){
      dim3 blockDim(ADDBLOCKXDIM);
      dim3 gridDim(size/ADDBLOCKXDIM + (size % ADDBLOCKXDIM == 0? 0:1));
      /**
       * Polynomial reduction
       */
      bn_t Q;
      get_words(&Q,q);
      polynomialReductionCoefs<<< gridDim,blockDim, 0, get_stream()>>>( d_bn_coefs,
                                                                        half,
                                                                        N,
                                                                        Q);
      cudaError_t result = cudaGetLastError();
      assert(result == cudaSuccess);
      
      set_crt_computed(false);
      set_icrt_computed(true);
      set_transf_computed(false);
      set_itransf_computed(false);
      set_host_updated(false);

      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);

      set_crt_computed(false);
      modn(q);
    }

    #endif
  }
}
