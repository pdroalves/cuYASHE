#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H
#include <cuda_runtime.h>
#include <stdint.h>
#include <assert.h>
#include "common.h"
#include <NTL/ZZ.h>

NTL_CLIENT

class CUDAFunctions{
  public:
  	static int N;
  	static cuyasheint_t wN;
	static cuyasheint_t *d_W;
	static cuyasheint_t *d_WInv;

    static cuyasheint_t* callPolynomialAddSub(cudaStream_t stream,cuyasheint_t *a,cuyasheint_t *b,int size,int OP);
    static cuyasheint_t* callPolynomialMul(cudaStream_t stream,cuyasheint_t *a,cuyasheint_t *b, int N, int NPolis);
    static cuyasheint_t* callRealignCRTResidues(cudaStream_t stream,int oldSpacing,int newSpacing, cuyasheint_t *array,int residuesSize,int residuesQty);
    static void init(int N){
    	
		#ifdef VERBOSE
		std::cout << "Will compute W."
		#endif

		CUDAFunctions::N = N;
		cuyasheint_t *h_W;
		cuyasheint_t *h_WInv;

		ZZ PZZ = conv<ZZ>("2147483647");
		cuyasheint_t k = conv<cuyasheint_t>(PZZ-1)/N;
		ZZ wNZZ = NTL::PowerMod(ZZ(7),k,PZZ);
		// assert((P-1)%(N) == 0);
		// const cuyasheint_t k = (P-1)/N;
		wN = conv<cuyasheint_t>(wNZZ);
		// wN = 17870292113338400769;

	    h_W = (cuyasheint_t*)malloc(N*N*sizeof(cuyasheint_t));
		cudaError_t result = cudaMalloc((void**)&d_W,N*N*sizeof(cuyasheint_t));
		assert(result == cudaSuccess);
		h_WInv = (cuyasheint_t*)malloc(N*N*sizeof(cuyasheint_t));
		result = cudaMalloc((void**)&d_WInv,N*N*sizeof(cuyasheint_t));
		assert(result == cudaSuccess);

		  // std::cout << "wN == " << wN << std::endl;
		  // std::cout << "k == " << k << std::endl;
		  // std::cout << "N == " << N << std::endl;
		  // std::cout << "P == " << PZZ << std::endl;

		// Computes W
		for(int j = 0; j < N; j++)
			for(int i = 0; i < N; i++)
			    // h_W[i+j*N] = (( j == 0)? 1:(h_W[i-1+j*N]*pow(wN,i)%q));
			    h_W[i+j*N] = conv<cuyasheint_t>(NTL::PowerMod(wNZZ,j*i,PZZ));

		for(int j = 0; j < N; j++)
			for(int i = 0; i < N; i++)
			    h_WInv[i+j*N] = conv<cuyasheint_t>(NTL::InvMod(conv<ZZ>(h_W[i+j*N]),PZZ ));

		result = cudaMemcpy(d_W,h_W , N*N*sizeof(cuyasheint_t), cudaMemcpyHostToDevice);
		assert(result == cudaSuccess);
		result = cudaMemcpy(d_WInv,h_WInv , N*N*sizeof(cuyasheint_t), cudaMemcpyHostToDevice);
		assert(result == cudaSuccess);

		free(h_W);
		free(h_WInv);
    }
  private:
};
cuyasheint_t s_rem (uint64_t a);
#endif
