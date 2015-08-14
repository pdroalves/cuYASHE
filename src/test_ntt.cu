#include <iostream>
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
  host_NTT(gridDim,blockDim,d_W,d_a,d_b,q,N,NPOLYS);
  assert(cudaGetLastError() == cudaSuccess);

  result = cudaMemset((void*)d_a,0,N*NPOLYS*sizeof(long));

  // Inverse
  host_NTT(gridDim,blockDim,d_WInv,d_b,d_a,q,N,NPOLYS);
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
