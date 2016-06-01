/**
 * cuYASHE
 * Copyright (C) 2015-2016 cuYASHE Authors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "settings.h"
#include "distribution.h"

__global__ void setup_kernel ( curandState * states, unsigned long seed )
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init ( seed, tid, 0, &states[tid] );
}

__host__ void Distribution::call_setup_kernel(){
	const int N = MAX_DEGREE;
	const int ADDGRIDXDIM = (N%ADDBLOCKXDIM == 0? N/ADDBLOCKXDIM : N/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

	setup_kernel<<<gridDim,blockDim,0>>>(states,SEED);
	assert(cudaGetLastError() == cudaSuccess);
}


__global__ void generate_random_numbers(bn_t* coefs, curandState *states, int N, int mod) {

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        coefs[tid].dp[0] = curand_uniform(&states[tid])*mod;
        coefs[tid].used = 1;
    }
}

__global__ void generate_random_numbers_CRT(cuyasheint_t* coefs, curandState *states, int N, int total, int mod) {

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        coefs[tid] = curand_uniform(&states[tid])*mod;
    }else{
    	if(tid < total)
	    	coefs[tid] = 0;
    }
}

__global__ void cuSetBNT(bn_t *a, cuyasheint_t *random,int N, int mod){
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < N){
		a[tid].used = 1;
		a[tid].dp[0] = random[tid] % mod;
		// bn_zero_non_used(&a[tid]);
	}
}

__host__  void Distribution::callCuGetUniformSample(cuyasheint_t* array, bn_t *coefs, int N, int mod){
	/**
	 * Generates N random integers
	 */

	// cuyasheint_t *random;
	// cudaMalloc((void**)&random,N*sizeof(cuyasheint_t));
	// curandStatus_t cuRandResult = curandGenerate( gen, 
	// 										(unsigned int*)(random), 
	// 										N);
	// assert(cuRandResult == CURAND_STATUS_SUCCESS);
 //  	/**
 //   	 * Sync
 //   	 */
	// cudaError_t result;
 //   	result = cudaDeviceSynchronize();// cuRAND doesn't use the same synchronization mechanism as others CUDAs APIs
 //  	assert(result == cudaSuccess);

  	/**
  	 * Adjust "used" attribute of each coefficient
  	 */
	const int ADDGRIDXDIM = (N%128 == 0? N/128 : N/128 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(128);

  	// cuSetBNT<<< gridDim, blockDim, 0>>>(coefs, random,N,mod);
  	// assert(cudaGetLastError() == cudaSuccess);
  	
  	// result = cudaFree(random);
  	// assert(cudaGetLastError() == cudaSuccess);
	

	/** 
	 * Generate values
	 */
	assert(N <= MAX_DEGREE);
	generate_random_numbers<<<gridDim,blockDim,0,0x0>>>(coefs,states,N,mod);
	assert(cudaGetLastError() == cudaSuccess);
}

__host__  void Distribution::callCuGetUniformSampleCRT(cuyasheint_t* array, int N, int NPolis, int mod){
	const int ADDGRIDXDIM = (N%128 == 0? N/128 : N/128 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(128);
	/** 
	 * Generate values
	 */
	assert(N <= MAX_DEGREE);
	cudaError_t result = cudaMemsetAsync(array,0,N*NPolis*sizeof(cuyasheint_t),0x0);
	assert(result == cudaSuccess);	
	generate_random_numbers_CRT<<<gridDim,blockDim,0,0x0>>>(array,states,N,N*NPolis,mod);
	assert(cudaGetLastError() == cudaSuccess);
}

__host__ void Distribution::callCuGetNormalSample(cuyasheint_t *array, int N, float mean, float stddev){
	///////////
	// To-do //
	///////////

	curandGenerateLogNormal( gen, 
							(float*)array, 
							N,
							mean,
							stddev);


}