#include "settings.h"
#include "distribution.h"

__global__ void cuAdjustBNT(bn_t *a, int N){
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < N)
		a[tid].used = 1;
}

__host__  void Distribution::callCuGetUniformSample(cuyasheint_t* array, bn_t *coefs, int N){
	/**
	 * Generates N random integers
	 */
	curandStatus_t cuRandResult = curandGenerate( gen, 
											(unsigned int*)(array), 
											N);
	assert(cuRandResult == CURAND_STATUS_SUCCESS);
  	
  	/**
  	 * Sync
  	 */
    cudaError_t result = cudaDeviceSynchronize();// cuRAND doesn't use the same synchronization mechanism as others CUDAs APIs
  	assert(result == cudaSuccess);

  	/**
  	 * Adjust "used" attribute of each coefficient
  	 */
	const int ADDGRIDXDIM = (N%ADDBLOCKXDIM == 0? N/ADDBLOCKXDIM : N/ADDBLOCKXDIM + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(ADDBLOCKXDIM);

  	cuAdjustBNT<<< gridDim, blockDim, 0>>>(coefs, N);
  	assert(cudaGetLastError() == cudaSuccess);
}

__host__ void Distribution::callCuGetNormalSample(cuyasheint_t *array, int N, float mean, float stddev){
	curandGenerateLogNormal( gen, 
							(float*)array, 
							N,
							mean,
							stddev);

}