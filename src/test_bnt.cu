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

	// Variables
	cuyasheint_t *a; 
	cuyasheint_t b;
	cuyasheint_t *result;

	// Settings
	unsigned int A_NBITS = 152;
	unsigned int A_NWORDS = (int)(A_NBITS/BN_DIGIT + (A_NBITS%BN_DIGIT != 0)*1 );

	// Init
	a = (cuyasheint_t*)malloc(A_NWORDS*sizeof(cuyasheint_t));
	// a*b has up to (A_NWORDS+1) words
	result = (cuyasheint_t*)malloc((A_NWORDS+1)*sizeof(cuyasheint_t));
	
	a[0] = 115206234L;
	a[1] = 1365961584L;
	a[2] = 2913851020L;
	a[3] = 3982751287L;
	a[4] = 9237030L;

	b = 3931036335;

	// ##############################################################################
	// Multiplication
	// ##############################################################################

	std::cout << "#######################################" << std::endl;
	std::cout << "#######################################" << std::endl;

	std::cout << "Multiplication" << std::endl;

	// Verify
	std::cout << "Checking operands:" << std::endl;
	for(unsigned int i =0; i < A_NWORDS; i++)
		std::cout << "a[" << i << "] = " << a[i] << std::endl;

	std::cout << "b = " << b << std::endl;

	// Multiply 
	int carry = bn_mul1_low(result, a, b, A_NWORDS);
	result[A_NWORDS] = carry;

	// Result
	std::cout << "Carry =  "<< carry << std::endl;
	std::cout << "Result: " << std::endl;
	for(unsigned int i =0; i <= A_NWORDS; i++)
		std::cout << "result[" << i << "] = " << result[i] << std::endl;

	free(result);
	// ##############################################################################
	// Addition
	// ##############################################################################

	std::cout << "#######################################" << std::endl;
	std::cout << "#######################################" << std::endl;
	// Variables
	cuyasheint_t *d;
	int D_NWORDS = A_NWORDS;

	// Init
	d = (cuyasheint_t*)malloc(D_NWORDS*sizeof(cuyasheint_t));
	result = (cuyasheint_t*)malloc(A_NWORDS*sizeof(cuyasheint_t));

	std::cout << "Addition" << std::endl;

	d[0] = 515944227L;
	d[1] = 1165891595L;
	d[2] = 259373817L;
	d[3] = 2703237026L;
	d[4] = 13L;

	std::cout << "Checking operands:" << std::endl;
	for(unsigned int i =0; i < A_NWORDS; i++)
		std::cout << "a[" << i << "] = " << a[i] << std::endl;
	for(unsigned int i =0; i < D_NWORDS; i++)
		std::cout << "d[" << i << "] = " << d[i] << std::endl;


	carry = bn_addn_low(result, a, d, A_NWORDS);

	std::cout << "Carry =  "<< carry << std::endl;
	std::cout << "Result: " << std::endl;
	for(unsigned int i =0; i < A_NWORDS; i++)
		std::cout << "result[" << i << "] = " << result[i] << std::endl;


	std::cout << "#######################################" << std::endl;
	std::cout << "#######################################" << std::endl;


	free(a);
	free(result);
	// ##############################################################################
	cudaError_t cuResult;

	cuResult = cudaDeviceSynchronize();
	assert(cuResult == cudaSuccess);
	/////////
	// CRT //
	/////////
	ZZ X = to_ZZ("5342748479728798479824982");
	bn_t *value;
	cudaMallocManaged((void**)&value,sizeof(bn_t));
	get_words(value,X);

	std::cout << "Input: " << std::endl;
	print(value);


    Polynomial::gen_crt_primes(to_ZZ("11728242269442118225223215137823427308012851928365899315927446782605336945293708352036740673805944948355911992121288532925023234127632779222593494093664877L"),1);
    CUDAFunctions::init(1);

	std::cout << "M: " << Polynomial::CRTProduct << std::endl;
	std::cout << "Mpi[0] =  " << Polynomial::CRTMpi[0] << std::endl;
	std::cout << "Mpi[0] words: " << std::endl;
	for(int i = 0; i < CUDAFunctions::Mpis[0].used;i++)
		std::cout << CUDAFunctions::Mpis[0].dp[i] << ", "<< std::endl;


	unsigned int nprimes = Polynomial::CRTPrimes.size();
    std::cout << nprimes << " primes generated" << std::endl;
    for(unsigned int i = 0; i < nprimes; i++)
    	std::cout << Polynomial::CRTPrimes[i] << ", " << std::endl;

	cuyasheint_t *d_polyCRT;
	cudaMalloc((void**)&d_polyCRT,nprimes*sizeof(cuyasheint_t));
	
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	crt(value,1,(cuyasheint_t*)d_polyCRT,1,nprimes,stream);

	/**
	 * Print residues for check
	 */
	
	cuyasheint_t *h_polyCRT;
	h_polyCRT = (cuyasheint_t*)malloc(nprimes*sizeof(cuyasheint_t));

	cudaMemcpy(h_polyCRT,d_polyCRT,nprimes*sizeof(cuyasheint_t),cudaMemcpyDeviceToHost);

	std::cout << std::endl << "Residues: " << std::endl;
	for(unsigned int i = 0; i < nprimes; i++)
		std::cout << h_polyCRT[i] << ", "<< std::endl;

	cuResult = cudaDeviceSynchronize();
	assert(cuResult == cudaSuccess);
	
	//////////
	// ICRT //
	//////////
	
	bn_zero(value);
	
	//
	//void icrt(bn_t *coefs,cuyasheint_t *d_polyCRT,const int N, const int NPolis,cudaStream_t stream){
	//
	icrt(value, (cuyasheint_t*)d_polyCRT,1,nprimes,stream);

	cuResult = cudaDeviceSynchronize();
	assert(cuResult == cudaSuccess);
	
	std::cout << std::endl << "ICRT result: " << std::endl;
	for(unsigned int i = 0; i < value->used; i++)
		std::cout << value->dp[i] << ", "<< std::endl;

	std::cout << std::endl << get_ZZ(value)%Polynomial::CRTProduct << " =? " << X << std::endl; 
	return 0;
}