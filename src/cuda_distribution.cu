#include "settings.h"
#include "distribution.h"

__global__ void setup_kernel ( curandState * states, unsigned long seed )
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init ( seed, tid, 0, &states[tid] );
}

__host__ void call_setup_kernel(curandState *states){
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
  	/**
   	 * Sync
   	 */
    // result = cudaDeviceSynchronize();// cuRAND doesn't use the same synchronization mechanism as others CUDAs APIs
  	// assert(result == cudaSuccess);

  	/**
  	 * Adjust "used" attribute of each coefficient
  	 */

  	// cuSetBNT<<< gridDim, blockDim, 0>>>(coefs, random,N,mod);
  	// assert(cudaGetLastError() == cudaSuccess);
  	
  	// result = cudaFree(random);
  	// assert(cudaGetLastError() == cudaSuccess);
	
	const int ADDGRIDXDIM = (N%ADDBLOCKXDIM == 0? N/ADDBLOCKXDIM : N/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);
	cudaError_t result;

	/** 
	 * Generate values
	 */
	assert(N <= MAX_DEGREE);
	generate_random_numbers<<<gridDim,blockDim,0,0x0>>>(coefs,states,N,mod);
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