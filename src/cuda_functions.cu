#include "cuda_functions.h"
#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>

///////////////////////////////////////
/// Memory operations
__global__ void realignCRTResidues(int oldSpacing,int newSpacing, long *array,long *new_array,int residuesSize,int residuesQty){
  //
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int residueId = tid / residuesSize;
  const int new_array_offset = (tid % residuesSize) + residueId*newSpacing;
  const int old_array_offset = (tid % residuesSize) + residueId*oldSpacing;

  if(new_array_offset < newSpacing*residuesQty && old_array_offset < oldSpacing*residuesQty )
    new_array[new_array_offset] = array[old_array_offset];

}


__host__ long* CUDAFunctions::callRealignCRTResidues(cudaStream_t stream,int oldSpacing,int newSpacing, long *array,int residuesSize,int residuesQty){
  if(oldSpacing == newSpacing)
    return NULL;
  const int size = residuesSize*residuesQty;
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  long *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,newSpacing*residuesQty*sizeof(long));
  #ifdef VERBOSE
  std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< newSpacing*residuesQty*sizeof(long) << " bytes" <<std::endl;
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
__global__ void polynomialAddSub(const int OP,const long *a,const long *b,long *c,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  long a_value;
  long b_value;

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

__host__ long* CUDAFunctions::callPolynomialAddSub(cudaStream_t stream,long *a,long *b,int size,int OP){
  // This method expects that both arrays are aligned
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  long *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,size*sizeof(long));
  #ifdef VERBOSE
  std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< size*sizeof(long) << " bytes" <<std::endl;
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

__global__ void polynomialPlainMul(const long *a,const long *b,long *c,const int N,const int NPolis){
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
  __shared__ long value;
  value = 0;

  // blockDim.x == blockDim.y
  // if(tidX < N && tidY < N){
    for(int tileY = 0;tileY < N/TILEDIM; tileY++)
      for(int tileX = 0;tileX < N/TILEDIM; tileX++){
        //      (           coefficient    ) + residue
        int i = (threadIdx.x + tileX*TILEDIM);
        int j = (threadIdx.y + tileY*TILEDIM);

        if(i + j == coefficient)
          atomicAdd((unsigned long long int*)(&value),(unsigned long long int)(a[i+residueOffset]*b[j+residueOffset]));
      }
    __syncthreads();

    if(threadIdx.x == 0 && threadIdx.y == 0)
      c[coefficient+residueOffset] = value;
      // There are 2N threads in Y axis computing this coefficient
      // atomicAdd((unsigned long long int*)(&(c[coefficient+residueOffset])),(unsigned long long int)(value));

  // }
}

__host__ long* CUDAFunctions::callPolynomialMul(cudaStream_t stream,long *a,long *b,int N,int NPolis){
  // This method expects that both arrays are aligned

  // Input:
  // stream: cudaStream
  // a: first operand
  // b: second operand
  // N: number of coefficients for each operand
  // NPolis: number of residues
  // All representations should be concatenated aligned
  assert((N>0)&&((N & (N - 1)) == 0));//Check if N is power of 2
  long *d_result;

  #ifdef PLAINMUL
    #ifdef VERBOSE
        std::cout << "Plain multiplication" << std::endl;
    #endif
    cudaError_t result = cudaMalloc((void**)&d_result,(2*N)*NPolis*sizeof(long));
    assert(result == cudaSuccess);
    result = cudaMemset((void*)d_result,0,(2*N)*NPolis*sizeof(long));
    assert(result == cudaSuccess);

    dim3 blockDim(ADDBLOCKXDIM,ADDBLOCKXDIM);
    // int blocks = ((2*N*NPolis) % ADDBLOCKXDIM == 0? (2*N*NPolis)/ADDBLOCKXDIM : (2*N*NPolis)/ADDBLOCKXDIM+1);
    // dim3 gridDim(blocks,blocks);
    dim3 gridDim(2*N*NPolis,1);
    polynomialPlainMul<<<gridDim,blockDim,1,stream>>>(a,b,d_result,N,NPolis);
    assert(cudaGetLastError() == cudaSuccess);
  #else

    // To-do
    throw "Polynomial multiplication not implemented!";
    // Allocates memory for temporary arrays on device
    // Each polynomial's degree gets doubled
    long *d_a;
    long *d_b;
    cudaError_t result = cudaMalloc((void**)&d_a,2*size*sizeof(long));
    assert(result == cudaSuccess);
    cudaError_t result = cudaMalloc((void**)&d_b,2*size*sizeof(long));
    assert(result == cudaSuccess);

    dim3 blockDim(32);
    dim3 gridDim((2*size)/32+1);
    NTT<<<gridDim,blockDim,1,stream>>>(a,d_a,N,size);
  #endif

  return d_result;
}
