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

__device__ void sumReduce(long value,long *a,int i,long q,int N, int NPolis){
  // Sum all elements in array "r" and writes to a, in position i

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  __shared__ long r[ADDBLOCKXDIM];
  r[threadIdx.x] = value;

  if(tid < N*NPolis){

    int stage = blockDim.x;
    while(stage > 0){// Equivalent to for(int i = 0; i < lrint(log2(N))+1;i++)
      if(threadIdx.x < stage/2 && (tid % N) + stage/2 < N){
        // Only half of the threads are used
        r[threadIdx.x] += r[threadIdx.x + stage/2];
      }
      stage /= 2;
      __syncthreads();
    }
    // After this loop, r[0] hold the sum of all block data

    if(threadIdx.x == 0)
      atomicAdd((unsigned long long int*)(&(a[i])),(unsigned long long int)(r[threadIdx.x] % q));
    __syncthreads();
  }
}

__global__ void NTT(long *W,long *a, long *a_hat, long q, int N,int NPolis){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;

  if(tid < N*NPolis){

    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      int cid = tid % N; // Coefficient id

      sumReduce(W[cid + i*N]*a[cid],a_hat,i,q,N,NPolis);
    }
  }

}

__global__ void INTT(){
  // This algorithm supposes that N is power of 2

}


__host__ void host_NTT(dim3 gridDim,dim3 blockDim,long *W,long *a, long *a_hat, long q,int N,int NPolis){
  // This is a dummy method used by the test framework. Probably unnecessary.
  NTT<<<gridDim,blockDim>>>(W,a,a_hat,q,N,NPolis);
}


__host__ long* CUDAFunctions::callPolynomialMul(cudaStream_t stream,long *a,long *b,int N,int NPolis){
  // All representations should be concatenated aligned
  assert((N>0)&&((N & (N - 1)) == 0));//Check if N is power of 2
  long *d_result;

  #ifdef PLAINMUL
    #ifdef VERBOSE
        std::cout << "Plain multiplication" << std::endl;
    #endif
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
