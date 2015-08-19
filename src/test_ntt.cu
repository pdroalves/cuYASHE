#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <iomanip>
#include <NTL/ZZ.h>
 #include <stdint.h>

#include "common.h"
#include "cuda_functions.h"

NTL_CLIENT

#define BILLION  1000000000L
#define MILLION  1000000L
#define DEGREE 64
#define NPOLYS 1
// #define P 0xffffffff00000001

double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}


__device__ void sumReduce(uint64_t value,uint64_t *a,int i,uint64_t q,int N, int NPolis){
  // Sum all elements in array "r" and writes to a, in position i

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  __shared__ uint64_t r[ADDBLOCKXDIM];
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
      atomicAdd((unsigned long long*)(&(a[i])),(unsigned long long)(r[threadIdx.x] % q));
    __syncthreads();
  }
}

__global__ void NTT(uint64_t *W,uint64_t *a, uint64_t *a_hat, int N,int NPolis){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const uint64_t p = 0xffffffff00000001;
  if(tid < N*NPolis){

    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      int cid = tid % N; // Coefficient id

      sumReduce(W[cid + i*N]*a[cid],a_hat,i,p,N,NPolis);
    }
  }

}

uint64_t powerMod(uint64_t x,long h,uint64_t p){
  unsigned long t;
  if(h == 0)
    return 1;
  else if(h == 1)
    return x % p;
  else
    t = log2((double)(h))+1;
  ZZ r = ZZ(x);
  ZZ X = ZZ(x);
  ZZ P = ZZ(p);

  for(int i = t-1; i >= 0; i--){
    r = r*r;
    r %= P;
    if((h >> i) & 1 == 1)//i-th bit
      r *= X % P;
    
  }
  return conv<uint64_t>(r);
}

int main(void){

  const int N = DEGREE;
  const uint64_t P = 0xffffffff00000001;
  assert((P-1)%(N) == 0);
  const uint64_t k = (P-1)/(N);
  const uint64_t wN = powerMod(7,k,P);
  // const uint64_t wN = 549755813888;// Hard coded
  std::cout << "wN == " << wN << std::endl;
  std::cout << "k == " << k << std::endl;
  std::cout << "N == " << N << std::endl;
  std::cout << "P == " << P << std::endl;
  // std::cout << "prime == " << prime << std::endl;
  // const uint64_t q = 97;

	dim3 blockDim(ADDBLOCKXDIM);
	dim3 gridDim((N*NPOLYS)/ADDBLOCKXDIM+1);

	uint64_t *h_a;
	uint64_t *d_a;
	uint64_t *h_b;
  uint64_t *d_b;
  uint64_t *h_W;
  uint64_t *d_W;
  uint64_t *h_WInv;
  uint64_t *d_WInv;

	// Alloc memory
	h_a = (uint64_t*)malloc(N*NPOLYS*sizeof(uint64_t));
	h_b = (uint64_t*)malloc(N*NPOLYS*sizeof(uint64_t));
  cudaError_t result = cudaMalloc((void**)&d_a,N*NPOLYS*sizeof(uint64_t));
	assert(result == cudaSuccess);
  result = cudaMalloc((void**)&d_b,N*NPOLYS*sizeof(uint64_t));
	assert(result == cudaSuccess);

  h_W = (uint64_t*)malloc(N*N*sizeof(uint64_t));
  result = cudaMalloc((void**)&d_W,N*N*sizeof(uint64_t));
	assert(result == cudaSuccess);
  h_WInv = (uint64_t*)malloc(N*N*sizeof(uint64_t));
  result = cudaMalloc((void**)&d_WInv,N*N*sizeof(uint64_t));
	assert(result == cudaSuccess);

  // Computes W
  for(int j = 0; j < N; j++)
    for(int i = 0; i < N; i++)
        // h_W[i+j*N] = (( j == 0)? 1:(h_W[i-1+j*N]*pow(wN,i)%q));
        h_W[i+j*N] = powerMod(wN,j*i,P);

  for(int j = 0; j < N; j++)
    for(int i = 0; i < N; i++)
        h_WInv[i+j*N] = powerMod(wN,-j*i,P);
  std::cout << "W computed. " << h_W[1+N] << " == " << wN << std::endl;
  uint64_t valor1 = pow(2,40);
  std::cout << "2**40 " << valor1  << std::endl;
  uint64_t valor2 = powerMod(2,40,P);
  std::cout << "valor2: "<< valor2 << std::endl;
  assert(h_W[1+N] == wN);
	// Generates random values
  for(int j = 0; j < NPOLYS;j++)
  	for(int i = 0; i < N/2; i++)
      h_a[i+j*NPOLYS] = i;
  		// h_a[i+j*NPOLYS] = rand() % q;

  // std::cout << "Input: " << std::endl;
  // for(int i = 0; i < N; i++)
  //   std::cout << h_a[i] << std::endl;

  // std::cout << "W: " << std::endl;

  // for(int i = 0; i < N; i++)
  //   std::cout << h_W[i] << std::endl;
  // for(int i = 0; i < N; i++)
  //   std::cout << h_W[i+1*N] << std::endl;

	// Copy to GPU
  result = cudaMemcpy(d_a,h_a , N*NPOLYS*sizeof(uint64_t), cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);

  result = cudaMemset((void*)d_b,0,N*NPOLYS*sizeof(uint64_t));

  result = cudaMemcpy(d_W,h_W , N*N*sizeof(uint64_t), cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);
  result = cudaMemcpy(d_WInv,h_WInv , N*N*sizeof(uint64_t), cudaMemcpyHostToDevice);
  assert(result == cudaSuccess);

	// Applies NTT
  // Foward
  NTT<<<gridDim,blockDim>>>(d_W,d_a,d_b,N,NPOLYS);
  assert(cudaGetLastError() == cudaSuccess);

  result = cudaMemset((void*)d_a,0,N*NPOLYS*sizeof(uint64_t));

  // Inverse
  NTT<<<gridDim,blockDim>>>(d_WInv,d_b,d_a,N,NPOLYS);
  assert(cudaGetLastError() == cudaSuccess);

	// Verify if the values were really shuffled
  result = cudaMemcpy(h_b,d_a,  N*NPOLYS*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	//
  // std::cout << "Output: " << std::endl;
  uint64_t NInv = NTL::InvMod(N,P);
  int count = 0;
  for(int i = 0; i < N; i++)
    if((h_b[i]*NInv % P) != h_a[i])
    std::cout << i << ") "<<(h_b[i]*NInv % P) << " != " << h_a[i] << std::endl;
      // count++;
  // std::cout << count << " errors." << std::endl;
	cudaFree(d_a);
	free(h_a);
	free(h_b);
  	std::cout << "Done." << std::endl;
}
