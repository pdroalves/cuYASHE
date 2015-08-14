#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <iomanip>
#include <NTL/ZZ.h>

#include "common.h"
#include "cuda_functions.h"

NTL_CLIENT

#define BILLION  1000000000L
#define MILLION  1000000L
#define N 32
#define NPOLYS 1

double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}


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

int main(void){

  const long wN = 8;// Hard coded
  const long q = 97;

	dim3 blockDim(ADDBLOCKXDIM);
	dim3 gridDim((N*NPOLYS)/ADDBLOCKXDIM+1);

	long *h_a;
	long *d_a;
	long *h_b;
  long *d_b;
  long *h_W;
  long *d_W;
  long *h_WInv;
  long *d_WInv;

	// Alloc memory
	h_a = (long*)malloc(N*NPOLYS*sizeof(long));
	h_b = (long*)malloc(N*NPOLYS*sizeof(long));
  cudaError_t result = cudaMalloc((void**)&d_a,N*NPOLYS*sizeof(long));
	assert(result == cudaSuccess);
  result = cudaMalloc((void**)&d_b,N*NPOLYS*sizeof(long));
	assert(result == cudaSuccess);

  h_W = (long*)malloc(N*N*sizeof(long));
  result = cudaMalloc((void**)&d_W,N*N*sizeof(long));
	assert(result == cudaSuccess);
  h_WInv = (long*)malloc(N*N*sizeof(long));
  result = cudaMalloc((void**)&d_WInv,N*N*sizeof(long));
	assert(result == cudaSuccess);

  // Computes W
  for(int j = 0; j < N; j++)
    for(int i = 0; i < N; i++)
        // h_W[i+j*N] = (( j == 0)? 1:(h_W[i-1+j*N]*pow(wN,i)%q));
        h_W[i+j*N] = NTL::PowerMod(wN,j*i,q);

  for(int j = 0; j < N; j++)
    for(int i = 0; i < N; i++)
        h_WInv[i+j*N] = NTL::PowerMod(wN,-j*i,q);
  std::cout << "W computed." << std::endl;
	// Generates random values
  for(int j = 0; j < NPOLYS;j++)
  	for(int i = 0; i < N/2; i++)
      h_a[i+j*NPOLYS] = i;
  		// h_a[i+j*NPOLYS] = rand() % q;

  std::cout << "Input: " << std::endl;
  for(int i = 0; i < N; i++)
    std::cout << h_a[i] << std::endl;

  // std::cout << "W: " << std::endl;

  // for(int i = 0; i < N; i++)
  //   std::cout << h_W[i] << std::endl;
  // for(int i = 0; i < N; i++)
  //   std::cout << h_W[i+1*N] << std::endl;

	// Copy to GPU
  result = cudaMemcpy(d_a,h_a , N*NPOLYS*sizeof(long), cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);

  result = cudaMemset((void*)d_b,0,N*NPOLYS*sizeof(long));

  result = cudaMemcpy(d_W,h_W , N*N*sizeof(long), cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);
  result = cudaMemcpy(d_WInv,h_WInv , N*N*sizeof(long), cudaMemcpyHostToDevice);
  assert(result == cudaSuccess);

	// Applies NTT
  // Foward
  NTT<<gridDim,blockDim>>>(d_W,d_a,d_b,q,N,NPOLYS);
  assert(cudaGetLastError() == cudaSuccess);

  result = cudaMemset((void*)d_a,0,N*NPOLYS*sizeof(long));

  // Inverse
  NTT<<gridDim,blockDim>>>(d_WInv,d_b,d_a,q,N,NPOLYS);
  assert(cudaGetLastError() == cudaSuccess);

	// Verify if the values were really shuffled
  result = cudaMemcpy(h_b,d_a,  N*NPOLYS*sizeof(long), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	//
  std::cout << "Output: " << std::endl;
  long NInv = NTL::InvMod(N,q);
  for(int i = 0; i < N; i++)
    std::cout << h_b[i]*NInv % q << " == " << h_a[i] << std::endl;

	cudaFree(d_a);
	free(h_a);
	free(h_b);
  	std::cout << "Done." << std::endl;
}
