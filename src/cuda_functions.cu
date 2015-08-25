#include "cuda_functions.h"
#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>

cuyasheint_t *(CUDAFunctions::d_W) = NULL;
cuyasheint_t *(CUDAFunctions::d_WInv) = NULL;
cuyasheint_t CUDAFunctions::wN = 0;
int CUDAFunctions::N = 0;

///////////////////////////////////////
/// Memory operations
__global__ void realignCRTResidues(int oldSpacing,int newSpacing, cuyasheint_t *array,cuyasheint_t *new_array,int residuesSize,int residuesQty){
  //
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int residueId = tid / residuesSize;
  const int new_array_offset = (tid % residuesSize) + residueId*newSpacing;
  const int old_array_offset = (tid % residuesSize) + residueId*oldSpacing;

  if(new_array_offset < newSpacing*residuesQty && old_array_offset < oldSpacing*residuesQty )
    new_array[new_array_offset] = array[old_array_offset];

}


__host__ cuyasheint_t* CUDAFunctions::callRealignCRTResidues(cudaStream_t stream,int oldSpacing,int newSpacing, cuyasheint_t *array,int residuesSize,int residuesQty){
  if(oldSpacing == newSpacing)
    return NULL;
  const int size = residuesSize*residuesQty;
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  cuyasheint_t *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,newSpacing*residuesQty*sizeof(cuyasheint_t));
  #ifdef VERBOSE
  std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< newSpacing*residuesQty*sizeof(cuyasheint_t) << " bytes" <<std::endl;
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
__global__ void polynomialAddSub(const int OP,const cuyasheint_t *a,const cuyasheint_t *b,cuyasheint_t *c,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  cuyasheint_t a_value;
  cuyasheint_t b_value;

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

__host__ cuyasheint_t* CUDAFunctions::callPolynomialAddSub(cudaStream_t stream,cuyasheint_t *a,cuyasheint_t *b,int size,int OP){
  // This method expects that both arrays are aligned
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  cuyasheint_t *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,size*sizeof(cuyasheint_t));
  #ifdef VERBOSE
  std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< size*sizeof(cuyasheint_t) << " bytes" <<std::endl;
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

__global__ void polynomialPlainMul(const cuyasheint_t *a,const cuyasheint_t *b,cuyasheint_t *c,const int N,const int NPolis){
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
  __shared__ cuyasheint_t value;
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
      // atomicAdd((unsigned cuyasheint_t cuyasheint_t int*)(&(c[coefficient+residueOffset])),(unsigned cuyasheint_t cuyasheint_t int)(value));

  // }
}


__device__ __host__ cuyasheint_t s_rem (uint64_t a)
{
  uint64_t res = (a>>31) + (a&0x7FFFFFFF);
  if(res > 0x7FFFFFFF)
    return (cuyasheint_t)((res>>31) + (res&0x7FFFFFFF));
  return (cuyasheint_t)res;
} 

__global__ void NTT64(cuyasheint_t *W,cuyasheint_t *WInv,cuyasheint_t *a, cuyasheint_t *a_hat, const int N,const int NPolis,const int type){
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


  if(tid < N*NPolis){
    // my_uint128 value = {0,0};
    cuyasheint_t value = 0;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      int64_t W64;
      if(type == FORWARD)
        W64 = W[i + cid*N];
      else
        W64 = WInv[i + cid*N];    

      int64_t a64 = a[i + roffset];
      value = s_rem((uint64_t)value + W64*a64);

    }
    if(type == FORWARD)
      a_hat[cid+roffset] = s_rem(value);
    else
      a_hat[cid+roffset] = s_rem(value)/N;

  }

}



__global__ void DOUBLENTT64( cuyasheint_t *W, cuyasheint_t *WInv,cuyasheint_t *a, cuyasheint_t *a_hat,cuyasheint_t *b, cuyasheint_t *b_hat, const int N,const int NPolis,const int type){
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
  // const cuyasheint_t P = 0xffffffff00000001;


  if(tid < N*NPolis){
    cuyasheint_t Avalue = 0;
    cuyasheint_t Bvalue = 0;
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
      Avalue = s_rem((uint64_t)Avalue + W64*a64);      
      Bvalue = s_rem((uint64_t)Bvalue + W64*b64);
    }
    if(type == FORWARD){
      a_hat[cid+ roffset] = s_rem(Avalue);
      b_hat[cid+ roffset] = s_rem(Bvalue);
    }else{
      a_hat[cid+ roffset] = s_rem(Avalue)/N;
      b_hat[cid+ roffset] = s_rem(Bvalue)/N;
    }
  }
}

__global__ void polynomialNTTMul(const cuyasheint_t *a,const cuyasheint_t *b,cuyasheint_t *c,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  // const cuyasheint_t P = 0xffffffff00000001;
  uint64_t a_value;
  uint64_t b_value;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = a[tid];
      b_value = b[tid];

      c[tid] = s_rem(a_value*b_value);
  }
}

__host__ cuyasheint_t* CUDAFunctions::callPolynomialMul(cudaStream_t stream,cuyasheint_t *a,cuyasheint_t *b,int N,int NPolis){
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
  cuyasheint_t *d_result;

  #ifdef PLAINMUL
    #ifdef VERBOSE
        std::cout << "Plain multiplication" << std::endl;
    #endif
    cudaError_t result = cudaMalloc((void**)&d_result,(N)*NPolis*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    result = cudaMemset((void*)d_result,0,(N)*NPolis*sizeof(cuyasheint_t));
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
    cuyasheint_t *d_a;
    cuyasheint_t *d_b;
    cuyasheint_t *d_c;
    cudaError_t result = cudaMalloc((void**)&d_a,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    // result = cudaMemset((void*)d_a,0,size*sizeof(cuyasheint_t));
    // assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_b,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    // result = cudaMemset((void*)d_b,0,size*sizeof(cuyasheint_t));
    // assert(result == cudaSuccess);    
    result = cudaMalloc((void**)&d_c,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_result,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    
    dim3 blockDim(ADDBLOCKXDIM);
    dim3 gridDim((size)/ADDBLOCKXDIM+1); // We expect that ADDBLOCKXDIM always divice size

    assert(blockDim.x*gridDim.x >= N);
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

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  #endif

  return d_result;
}
