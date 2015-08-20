#include "cuda_functions.h"
#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>

uint32_t *(CUDAFunctions::d_W) = NULL;
uint32_t *(CUDAFunctions::d_WInv) = NULL;
uint64_t CUDAFunctions::wN = 0;

///////////////////////////////////////
/// Memory operations
__global__ void realignCRTResidues(int oldSpacing,int newSpacing, uint32_t *array,uint32_t *new_array,int residuesSize,int residuesQty){
  //
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int residueId = tid / residuesSize;
  const int new_array_offset = (tid % residuesSize) + residueId*newSpacing;
  const int old_array_offset = (tid % residuesSize) + residueId*oldSpacing;

  if(new_array_offset < newSpacing*residuesQty && old_array_offset < oldSpacing*residuesQty )
    new_array[new_array_offset] = array[old_array_offset];

}


__host__ uint32_t* CUDAFunctions::callRealignCRTResidues(cudaStream_t stream,int oldSpacing,int newSpacing, uint32_t *array,int residuesSize,int residuesQty){
  if(oldSpacing == newSpacing)
    return NULL;
  const int size = residuesSize*residuesQty;
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  uint32_t *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,newSpacing*residuesQty*sizeof(uint32_t));
  #ifdef VERBOSE
  std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< newSpacing*residuesQty*sizeof(uint32_t) << " bytes" <<std::endl;
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
__global__ void polynomialAddSub(const int OP,const uint32_t *a,const uint32_t *b,uint32_t *c,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  uint32_t a_value;
  uint32_t b_value;

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

__host__ uint32_t* CUDAFunctions::callPolynomialAddSub(cudaStream_t stream,uint32_t *a,uint32_t *b,int size,int OP){
  // This method expects that both arrays are aligned
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  uint32_t *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,size*sizeof(uint32_t));
  #ifdef VERBOSE
  std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< size*sizeof(uint32_t) << " bytes" <<std::endl;
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

__global__ void polynomialPlainMul(const uint32_t *a,const uint32_t *b,uint32_t *c,const int N,const int NPolis){
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
  __shared__ uint32_t value;
  value = 0;

  // blockDim.x == blockDim.y
  // if(tidX < N && tidY < N){
    for(int tileY = 0;tileY < N/TILEDIM; tileY++)
      for(int tileX = 0;tileX < N/TILEDIM; tileX++){
        //      (           coefficient    ) + residue
        int i = (threadIdx.x + tileX*TILEDIM);
        int j = (threadIdx.y + tileY*TILEDIM);

        if(i + j == coefficient)
          atomicAdd((&value),(a[i+residueOffset]*b[j+residueOffset]));
      }
    __syncthreads();

    if(threadIdx.x == 0 && threadIdx.y == 0)
      c[coefficient+residueOffset] = value;
      // There are 2N threads in Y axis computing this coefficient
      // atomicAdd((unsigned uint32_t uint32_t int*)(&(c[coefficient+residueOffset])),(unsigned uint32_t uint32_t int)(value));

  // }
}

__global__ void NTT32(uint32_t *W,uint32_t *WInv,uint32_t *a, uint32_t *a_hat, const int N,const int NPolis,const uint64_t P,const int type){
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
  // const double invk = (double)(1<<30) / P;
  uint32_t *w;
  if(type == FORWARD)
    w = W;
  else
    w = WInv;

  // const inteiro p = 0xffffffff00000001;
  if(tid < N*NPolis){
    uint64_t value = 0;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      value = (value + W64*a64)%P;
      // value = value + mul_m(W64,a64,P,invk);
    }
    if(type == FORWARD)
      a_hat[cid+roffset] = value % P;
    else
      a_hat[cid+roffset] = (value % P)/N;

  }

}



__global__ void DOUBLENTT32( uint32_t *W, uint32_t *WInv,uint32_t *a, uint32_t *a_hat,uint32_t *b, uint32_t *b_hat, const int N,const int NPolis,const uint64_t P,const int type){
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
  uint32_t *w;
  if(type == FORWARD)
    w = W;
  else
    w = WInv;

  // const uint32_t p = 0xffffffff00000001;
  if(tid < N*NPolis){
    uint64_t Avalue = 0;
    uint64_t Bvalue = 0;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){

      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      uint64_t b64 = b[i + roffset];
      Avalue = (Avalue + W64*a64)%P;      
      Bvalue = (Bvalue + W64*b64)%P;
    }
    if(type == FORWARD){
      a_hat[cid+ roffset] = Avalue % P;
      b_hat[cid+ roffset] = Bvalue % P;
    }else{
      a_hat[cid+ roffset] = (Avalue % P)/N;
      b_hat[cid+ roffset] = (Bvalue % P)/N;
    }
  }

}

__global__ void polynomialNTTMul(const uint32_t *a,const uint32_t *b,uint32_t *c,const int size,const uint64_t P){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  uint64_t a_value;
  uint64_t b_value;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = a[tid];
      b_value = b[tid];

      a_value = (a_value*b_value) % P;

      c[tid] = a_value;
  }
}

__host__ uint32_t* CUDAFunctions::callPolynomialMul(cudaStream_t stream,uint32_t *a,uint32_t *b,int N,int NPolis){
  // This method expects that both arrays are aligned

  // Input:
  // stream: cudaStream
  // a: first operand
  // b: second operand
  // N: number of coefficients for each operand
  // NPolis: number of residues
  // All representations should be concatenated aligned
  assert((N>0)&&((N & (N - 1)) == 0));//Check if N is power of 2
  uint32_t *d_result;

  #ifdef PLAINMUL
    #ifdef VERBOSE
        std::cout << "Plain multiplication" << std::endl;
    #endif
    cudaError_t result = cudaMalloc((void**)&d_result,(2*N)*NPolis*sizeof(uint32_t));
    assert(result == cudaSuccess);
    result = cudaMemset((void*)d_result,0,(2*N)*NPolis*sizeof(uint32_t));
    assert(result == cudaSuccess);

    dim3 blockDim(ADDBLOCKXDIM,ADDBLOCKXDIM);
    // int blocks = ((2*N*NPolis) % ADDBLOCKXDIM == 0? (2*N*NPolis)/ADDBLOCKXDIM : (2*N*NPolis)/ADDBLOCKXDIM+1);
    // dim3 gridDim(blocks,blocks);
    dim3 gridDim(2*N*NPolis,1);
    polynomialPlainMul<<<gridDim,blockDim,1,stream>>>(a,b,d_result,N,NPolis);
    assert(cudaGetLastError() == cudaSuccess);
  #elif defined(NTTMUL)
        std::cout << "NTT multiplication" << std::endl;

    // Allocates memory for temporary arrays on device
    // Each polynomial's degree gets doubled
    const int size = N*NPolis;
    uint32_t *d_a;
    uint32_t *d_b;
    uint32_t *d_c;
    cudaError_t result = cudaMalloc((void**)&d_a,size*sizeof(uint32_t));
    assert(result == cudaSuccess);
    result = cudaMemset((void*)d_a,0,size*sizeof(uint32_t));
    assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_b,size*sizeof(uint32_t));
    assert(result == cudaSuccess);
    result = cudaMemset((void*)d_b,0,size*sizeof(uint32_t));
    assert(result == cudaSuccess);    
    result = cudaMalloc((void**)&d_c,size*sizeof(uint32_t));
    assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_result,size*NPolis*sizeof(uint32_t));
    assert(result == cudaSuccess);
    
    dim3 blockDim(ADDBLOCKXDIM);
    dim3 gridDim((size)/ADDBLOCKXDIM); // We expect that ADDBLOCKXDIM always divice size

    // Forward 
    DOUBLENTT32<<<gridDim,blockDim,1,stream>>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,a,d_a,b,d_b,N,NPolis,CUDAFunctions::P,FORWARD);
    assert(cudaGetLastError() == cudaSuccess);

    // Multiply
    polynomialNTTMul<<<gridDim,blockDim,1,stream>>>(d_a,d_b,d_c,N*NPolis,P);

    // Inverse    
    NTT32<<<gridDim,blockDim,1,stream>>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,d_c,d_result,N,NPolis,CUDAFunctions::P,INVERSE);
    assert(cudaGetLastError() == cudaSuccess);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
  #endif

  return d_result;
}
