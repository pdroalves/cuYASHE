#ifndef CUDA_CIPHERTEXT_H
#define CUDA_CIPHERTEXT_H
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_bn.h"
#include "polynomial.h"

template <int WORDLENGTH = 32>
extern __global__ void cuWordecomp(bn_t **P,bn_t *a,int lwq, int N);
void callCuWordecomp(dim3 gridDim, dim3 blockDim, cudaStream_t stream, int WORDLENGTH, bn_t **d_P, bn_t *a, int lwq, int N);
__host__ __device__ void convert_64_to_32(uint32_t *a,uint64_t *b,int n);
__host__ __device__ void convert_32_to_64(uint64_t *a, uint32_t *b, int n);

template <int WORDLENGTH>
__host__ void callWordDecomp(	std::vector<Polynomial> *P,
								bn_t *a,
								int lwq,
								int N,
								bn_t W,
								bn_t u_W,
								cudaStream_t stream
							){
	cudaError_t result;

	/**
	 * P is a collection of lwq arrays of size (deg+1)
	 */

	// Alloc memory for P
	bn_t **d_P;
	result = cudaMalloc((void**)&d_P,N*sizeof(bn_t*));
	assert(result == cudaSuccess);

	bn_t **h_P;
	h_P = (bn_t**)malloc(N*sizeof(bn_t*));

	for(int i = 0; i < P->size(); i++)
		h_P[i] = P->at(i).d_bn_coefs;

	result = cudaMemcpyAsync(d_P,h_P,N*sizeof(bn_t*),cudaMemcpyHostToDevice,stream);
	assert(result == cudaSuccess);

	// Worddecomp
	const int ADDGRIDXDIM = (N%180 == 0? N/180 : N/180 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(180);

	callCuWordecomp(gridDim,blockDim,stream,WORDLENGTH,d_P,a,lwq, N);

	for(int i = 0; i < P->size(); i++){
		P->at(i).set_icrt_computed(true);
		P->at(i).set_crt_computed(false);
		P->at(i).set_host_updated(false);
	}

	result = cudaDeviceSynchronize();
	assert(result == cudaSuccess);
	

	free(h_P);
	result = cudaFree(d_P);
	assert(result == cudaSuccess);
}


__host__ void callCiphertextMulAux(	bn_t *P, 
									bn_t *g, 
									ZZ q,
									int N, 
									cudaStream_t stream);
__host__ void callCiphertextMulAuxMersenne(bn_t *g, ZZ q,int N, cudaStream_t stream);

#endif