#include "cuda_functions.h"
#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "modop.h"

uint64_t *(CUDAFunctions::d_W) = NULL;
uint64_t *(CUDAFunctions::d_WInv) = NULL;
uint64_t CUDAFunctions::wN = 0;
int CUDAFunctions::N = 0;

///////////////////////////////////////
/// Memory operations
__global__ void realignCRTResidues(int oldSpacing,int newSpacing, uint64_t *array,uint64_t *new_array,int residuesSize,int residuesQty){
  //
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int residueId = tid / residuesSize;
  const int new_array_offset = (tid % residuesSize) + residueId*newSpacing;
  const int old_array_offset = (tid % residuesSize) + residueId*oldSpacing;

  if(new_array_offset < newSpacing*residuesQty && old_array_offset < oldSpacing*residuesQty )
    new_array[new_array_offset] = array[old_array_offset];

}


__host__ uint64_t* CUDAFunctions::callRealignCRTResidues(cudaStream_t stream,int oldSpacing,int newSpacing, uint64_t *array,int residuesSize,int residuesQty){
  if(oldSpacing == newSpacing)
    return NULL;
  const int size = residuesSize*residuesQty;
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  uint64_t *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,newSpacing*residuesQty*sizeof(uint64_t));
  #ifdef VERBOSE
  std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< newSpacing*residuesQty*sizeof(uint64_t) << " bytes" <<std::endl;
  #endif
  assert(result == cudaSuccess);

  realignCRTResidues <<< gridDim,blockDim,1,stream >>> (oldSpacing,newSpacing,array,d_new_array,residuesSize,residuesQty);
  #ifdef VERBOSE
  std::cout << gridDim.x << " " << blockDim.x << std::endl;
  std::cout << "polynomialAdd kernel:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
  #endif
  assert(cudaGetLastError() == cudaSuccess);

  return d_new_array;
}

///////////////////////////////////////

///////////////////////////////////////
/// ADD
__global__ void polynomialAddSub(const int OP,const uint64_t *a,const uint64_t *b,uint64_t *c,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  uint64_t a_value;
  uint64_t b_value;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = a[tid];
      b_value = b[tid];

      if(OP == ADD)
        a_value += b_value;
      else
        a_value -= b_value;

      c[tid] = a_value;
  }
}

__host__ uint64_t* CUDAFunctions::callPolynomialAddSub(cudaStream_t stream,uint64_t *a,uint64_t *b,int size,int OP){
  // This method expects that both arrays are aligned
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  uint64_t *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,size*sizeof(uint64_t));
  #ifdef VERBOSE
  std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< size*sizeof(uint64_t) << " bytes" <<std::endl;
  #endif
  assert(result == cudaSuccess);

  // polynomialAdd <<< gridDim,blockDim, 0, stream >>> (a,b,d_new_array,size);
  polynomialAddSub <<< gridDim,blockDim >>> (OP,a,b,d_new_array,size);
  #ifdef VERBOSE
  std::cout << gridDim.x << " " << blockDim.x << std::endl;
  std::cout << "polynomialAdd kernel:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
  #endif

  return d_new_array;
}
///////////////////////////////////////

///////////////////////////////////////
/// MUL

__global__ void polynomialPlainMul(const uint64_t *a,const uint64_t *b,uint64_t *c,const int N,const int NPolis){
  // Each block computes one coefficient of c
  // We need 2*N blocks for each residue!
  // 2D-blocks

  // const int tidX = threadIdx.x + blockDim.x*blockIdx.x;
  // const int tidY = threadIdx.y + blockDim.y*blockIdx.y;

  // blockDim.x == blockDim.y
  // We suppose that TILEDIM divides 2*N
  const int TILEDIM = blockDim.x;
  const int coefficient = blockIdx.x % (2*N);
  const int residueID = blockIdx.x / (2*N);
  const int residueOffset = residueID*(2*N);
  __shared__ uint64_t value;
  value = 0;

  // blockDim.x == blockDim.y
  // if(tidX < N && tidY < N){
    for(int tileY = 0;tileY < N/TILEDIM; tileY++)
      for(int tileX = 0;tileX < N/TILEDIM; tileX++){
        //      (           coefficient    ) + residue
        int i = (threadIdx.x + tileX*TILEDIM);
        int j = (threadIdx.y + tileY*TILEDIM);

        if(i + j == coefficient)
          atomicAdd((unsigned long long int *)(&value),(unsigned long long int)(a[i+residueOffset]*b[j+residueOffset]));
      }
    __syncthreads();

    if(threadIdx.x == 0 && threadIdx.y == 0)
      c[coefficient+residueOffset] = value;
      // There are 2N threads in Y axis computing this coefficient
      // atomicAdd((unsigned uint64_t uint64_t int*)(&(c[coefficient+residueOffset])),(unsigned uint64_t uint64_t int)(value));

  // }
}

__device__ uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t res = 0;
    uint64_t temp_b;

    while (a != 0) {
        if (a & 1) {
            /* Add b to res, modulo m, without overflow */
            if (b >= m - res) /* Equiv to if (res + b >= m), without overflow */
                res -= m;
            res += b;
        }
        a >>= 1;

        /* Double b, modulo m */
        temp_b = b;
        if (b >= m - b)       /* Equiv to if (2 * b >= m), without overflow */
            temp_b -= m;
        b += temp_b;
    __syncthreads();
    }
    return res;
}

typedef struct {
  unsigned long long int lo;
  unsigned long long int hi;
} my_uint128;

__device__ my_uint128 add_uint128 (my_uint128 a, my_uint128 b)
{
  my_uint128 res;
  res.lo = a.lo + b.lo;
  res.hi = a.hi + b.hi + (res.lo < a.lo);
  return res;
} 

__device__ my_uint128 add_uint64_128 (uint64_t a, my_uint128 b)
{
  my_uint128 res = {a+b.lo,b.hi}; // FALHA SE A+B.LO > 64 BITS

  // my_uint128 res;
  // res.lo = a + b.lo;
  // res.hi = b.hi + (res.lo < a);
  return res;
} 

__device__ my_uint128 sub_uint128_64 (my_uint128 a, uint64_t b)
{
  // a - b

  // my_uint128 res;
  // res.lo = a + b.lo;
  // res.hi = b.hi + (res.lo < a);
  
  my_uint128 res;
  const u_int64_t borrow = b > a.lo;

  res.lo = a.lo - b;
  res.hi = a.hi - borrow;

  return res;
} 


__device__ my_uint128 add_uint64_64 (uint64_t a, uint64_t b)
{

  my_uint128 res;
  res.lo = a + b;
  res.hi = (res.lo < a);

  return res;
} 

__device__ my_uint128 mul_uint64_128 (uint64_t a, uint64_t b)
{
  my_uint128 res = {a*b,__umul64hi(a,b)};
  return res;
} 





__device__ uint64_t s_rem (uint64_t a)
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
  uint64_t x1 = (a >> 32); // Max 32 bits
  uint64_t x0 = (a & UINT32_MAX); // Max 32 bits

  // uint64_t res = ((x1+x2)<<32 + x0-x3-x2);
  uint64_t res = ((x1<<32) + x0); // Max 64 bits

  return res;
} 

__device__ uint64_t s_rem (my_uint128 a)
{
  // Special reduction for prime 2^64-2^32+1
  //
  // x3 * 2^96 + x2 * 2^64 + x1 * 2^32 + x0 \equiv
  // (x1+x2) * 2^32 + x0 - x3 - x2 mod p
  //
  // Here: x3 = 0, x2 = a.hi, x1 = (a.lo >> 32), x0 = a.lo-(x1 << 32)
  // const uint64_t p = 0xffffffff00000001;
  uint64_t x3 = (a.hi >> 32); // Max 32 bits
  uint64_t x2 = (a.hi & UINT32_MAX); // Max 32 bits
  uint64_t x1 = (a.lo >> 32); // Max 32 bits
  uint64_t x0 = (a.lo & UINT32_MAX); // Max 32 bits

  // my_uint128 x1Px232StL = {x0,(x1+x2)}; // x1 plus x2 32 Shift to Left

  // uint64_t res = sub_uint128_64(sub_uint128_64(x1Px232StL,x3),x2).lo; // -x3

  // uint64_t res = (x1 << 32);
  // res -= (x3+x2);
  // res += (x2 << 32)+x0;

  uint64_t res = ((x1+x2)<<32)+x0-x2-x3;
  return res;
} 

__global__ void NTT64(uint64_t *W,uint64_t *WInv,uint64_t *a, uint64_t *a_hat, const int N,const int NPolis,const int type){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues
   const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int residueid = tid / (N);
  const int roffset = residueid*N;
  const int cid = tid & (N-1); // Coefficient id
  // const uint64_t P = 0xffffffff00000001;


  if(tid < N*NPolis){
    // my_uint128 value = {0,0};
    uint64_t value = 0;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      uint64_t W64;
      if(type == FORWARD)
        W64 = W[i + cid*N];
      else
        W64 = WInv[i + cid*N];    

      uint64_t a64 = a[i + roffset];
      // value = (add_uint128(value, mul_uint64_128(W64,a64)));
      value = (mod_add(value, mod_mul(W64,a64)));

    }
    if(type == FORWARD)
      a_hat[cid+roffset] = (value);
    else
      a_hat[cid+roffset] = (value)/N;

  }

}



__global__ void DOUBLENTT64( uint64_t *W, uint64_t *WInv,uint64_t *a, uint64_t *a_hat,uint64_t *b, uint64_t *b_hat, const int N,const int NPolis,const int type){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int residueid = tid / N;
  const int roffset = residueid*N;
  const int cid = tid & (N-1); // Coefficient id
  // const uint64_t P = 0xffffffff00000001;


  if(tid < N*NPolis){
    uint64_t Avalue = 0;
    uint64_t Bvalue = 0;
    // my_uint128 Avalue = {0,0};
    // my_uint128 Bvalue = {0,0};
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      uint64_t W64;
      if(type == FORWARD)
        W64 = W[i + cid*N];
      else
        W64 = WInv[i + cid*N];

      uint64_t a64 = a[i + roffset];
      uint64_t b64 = b[i + roffset];
      // Avalue = (add_uint128(Avalue, mul_uint64_128(W64,a64)));      
      // Bvalue = (add_uint128(Bvalue, mul_uint64_128(W64,b64)));
      Avalue = (mod_add(Avalue, mod_mul(W64,a64)));      
      Bvalue = (mod_add(Bvalue, mod_mul(W64,b64)));
    }
    if(type == FORWARD){
      a_hat[cid+ roffset] = (Avalue);
      b_hat[cid+ roffset] = (Bvalue);
    }else{
      a_hat[cid+ roffset] = (Avalue)/N;
      b_hat[cid+ roffset] = (Bvalue)/N;
    }
  }
}

__global__ void polynomialNTTMul(const uint64_t *a,const uint64_t *b,uint64_t *c,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  // const uint64_t P = 0xffffffff00000001;
  uint64_t a_value;
  uint64_t b_value;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = a[tid];
      b_value = b[tid];

      // a_value = s_rem(mul_uint64_128(a_value,b_value));
      a_value = mod_mul(a_value,b_value);

      c[tid] = a_value;
  }
}

__host__ uint64_t* CUDAFunctions::callPolynomialMul(cudaStream_t stream,uint64_t *a,uint64_t *b,int N,int NPolis){
  // This method expects that both arrays are aligned

  // Input:
  // stream: cudaStream
  // a: first operand
  // b: second operand
  // N: number of coefficients for each operand
  // NPolis: number of residues
  // All representations should be concatenated aligned
  assert((N>0)&&((N & (N - 1)) == 0));//Check if N is power of 2
  assert(N == CUDAFunctions::N);
  uint64_t *d_result;

  #ifdef PLAINMUL
    #ifdef VERBOSE
        std::cout << "Plain multiplication" << std::endl;
    #endif
    cudaError_t result = cudaMalloc((void**)&d_result,(N)*NPolis*sizeof(uint64_t));
    assert(result == cudaSuccess);
    result = cudaMemset((void*)d_result,0,(N)*NPolis*sizeof(uint64_t));
    assert(result == cudaSuccess);

    dim3 blockDim(ADDBLOCKXDIM,ADDBLOCKXDIM);
    // int blocks = ((2*N*NPolis) % ADDBLOCKXDIM == 0? (2*N*NPolis)/ADDBLOCKXDIM : (2*N*NPolis)/ADDBLOCKXDIM+1);
    // dim3 gridDim(blocks,blocks);
    dim3 gridDim(N*NPolis,1);
    polynomialPlainMul<<<gridDim,blockDim,1,stream>>>(a,b,d_result,N,NPolis);
    assert(cudaGetLastError() == cudaSuccess);
  #elif defined(NTTMUL)
        // std::cout << "NTT multiplication" << std::endl;

    // Allocates memory for temporary arrays on device
    // Each polynomial's degree gets doubled
    const int size = N*NPolis;
    uint64_t *d_a;
    uint64_t *d_b;
    uint64_t *d_c;
    cudaError_t result = cudaMalloc((void**)&d_a,size*sizeof(uint64_t));
    assert(result == cudaSuccess);
    // result = cudaMemset((void*)d_a,0,size*sizeof(uint64_t));
    // assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_b,size*sizeof(uint64_t));
    assert(result == cudaSuccess);
    // result = cudaMemset((void*)d_b,0,size*sizeof(uint64_t));
    // assert(result == cudaSuccess);    
    result = cudaMalloc((void**)&d_c,size*sizeof(uint64_t));
    assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_result,size*sizeof(uint64_t));
    assert(result == cudaSuccess);
    
    dim3 blockDim(ADDBLOCKXDIM);
    dim3 gridDim((size)/ADDBLOCKXDIM); // We expect that ADDBLOCKXDIM always divice size

    // Forward 
    DOUBLENTT64<<<gridDim,blockDim,1,stream>>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,a,d_a,b,d_b,N,NPolis,FORWARD);
    assert(cudaGetLastError() == cudaSuccess);

    // Multiply
    polynomialNTTMul<<<gridDim,blockDim,1,stream>>>(d_a,d_b,d_c,N*NPolis);

    // Inverse    
    NTT64<<<gridDim,blockDim,1,stream>>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,d_c,d_result,N,NPolis,INVERSE);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
    assert(cudaGetLastError() == cudaSuccess);

    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  #endif

  return d_result;
}
