#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H
#include <cuda_runtime.h>
#include <stdint.h>
#include <assert.h>
#include <NTL/ZZ.h>

NTL_CLIENT
// void host_bitreverse(dim3 gridDim,dim3 blockDim,long *a,int n,int npolys);
// void host_NTT(dim3 gridDim,dim3 blockDim,long *W,long *a, long *a_hat, long q,int N,int NPolis);

class CUDAFunctions{
  public:

	static uint32_t *d_W;
	static uint32_t *d_WInv;
	const static uint64_t P = 4294955009;//31 bits

    static uint32_t* callPolynomialAddSub(cudaStream_t stream,uint32_t *a,uint32_t *b,int size,int OP);
    static uint32_t* callPolynomialMul(cudaStream_t stream,uint32_t *a,uint32_t *b, int N, int NPolis);
    static uint32_t* callRealignCRTResidues(cudaStream_t stream,int oldSpacing,int newSpacing, uint32_t *array,int residuesSize,int residuesQty);
    static void init(int N){

			uint32_t *h_W;
			uint32_t *h_WInv;

			// const uint64_t P = 4294955009;//31 bits
			assert((P-1)%(N) == 0);
			const uint64_t k = (P-1)/N;
			const uint64_t wN = NTL::PowerMod(3,k,P);

		    h_W = (uint32_t*)malloc(N*N*sizeof(uint32_t));
			cudaError_t result = cudaMalloc((void**)&d_W,N*N*sizeof(uint32_t));
			assert(result == cudaSuccess);
			h_WInv = (uint32_t*)malloc(N*N*sizeof(uint32_t));
			result = cudaMalloc((void**)&d_WInv,N*N*sizeof(uint32_t));
			assert(result == cudaSuccess);

			// Computes W
			for(int j = 0; j < N; j++)
				for(int i = 0; i < N; i++)
				    // h_W[i+j*N] = (( j == 0)? 1:(h_W[i-1+j*N]*pow(wN,i)%q));
				    h_W[i+j*N] = NTL::PowerMod(wN,j*i,P);

			for(int j = 0; j < N; j++)
				for(int i = 0; i < N; i++)
				    h_WInv[i+j*N] = NTL::InvMod(h_W[i+j*N],P);

			result = cudaMemcpy(d_W,h_W , N*N*sizeof(uint32_t), cudaMemcpyHostToDevice);
			assert(result == cudaSuccess);
			result = cudaMemcpy(d_WInv,h_WInv , N*N*sizeof(uint32_t), cudaMemcpyHostToDevice);
			assert(result == cudaSuccess);

			free(h_W);
			free(h_WInv);
    }
  private:
};

#endif
