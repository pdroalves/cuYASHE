#include "settings.h"
#include "cuda_bn.h"
#include "polynomial.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>


__global__ void kernel(bn_t *value){

	for(int word = 0; word < value->used; word++)
		printf("GPU: %u\n",value->dp[word]);

	return;
}

void print(bn_t *value){
	for(int word = 0; word < value->used; word++)
		std::cout << "CPU: " << value->dp[word] << std::endl;

	std::cout << std::endl;
	cudaDeviceSynchronize();
	kernel<<<1,1>>>(value);
}

int main(void){
	bn_t *value;
	cudaMallocManaged((void**)&value,sizeof(bn_t));

	bn_new(value);
	bn_grow(value,3);

	value->dp[0] = 2497329238L;
	value->dp[1] = 4189475509L;
	value->dp[2] = 289630L;
	value->used = 3;

	std::cout << "Input: " << std::endl;
	print(value);

	/////////
	// CRT //
	/////////

    Polynomial::gen_crt_primes(to_ZZ("11728242269442118225223215137823427308012851928365899315927446782605336945293708352036740673805944948355911992121288532925023234127632779222593494093664877L"),1);
    CUDAFunctions::init(1);

	unsigned int nprimes = Polynomial::CRTPrimes.size();
    std::cout << nprimes << " primes generated" << std::endl;
    for(unsigned int i = 0; i < nprimes; i++)
    	std::cout << Polynomial::CRTPrimes[i] << ", " << std::endl;

	cuyasheint_t *d_polyCRT;
	cudaMalloc((void**)&d_polyCRT,nprimes*sizeof(cuyasheint_t));
	
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	crt(value,d_polyCRT,1,nprimes,stream);

	/**
	 * Check if the residues are correct
	 */
	
	cuyasheint_t *h_polyCRT;
	h_polyCRT = (cuyasheint_t*)malloc(nprimes*sizeof(cuyasheint_t));

	cudaMemcpy(h_polyCRT,d_polyCRT,nprimes*sizeof(cuyasheint_t),cudaMemcpyDeviceToHost);

	std::cout << std::endl << "Residues: " << std::endl;
	for(unsigned int i = 0; i < nprimes; i++)
		std::cout << h_polyCRT[i] << ", "<< std::endl;

	cudaDeviceSynchronize();

	return 0;
}