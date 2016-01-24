#include "settings.h"
#include "distribution.h"

__host__  void Distribution::callCuGetUniformSample(cuyasheint_t*array, int N){
	/**
	 * Generates N*NPolis random integers
	 */
	curandStatus_t result = curandGenerate( gen, 
											(unsigned int*)(array), 
											N);
	assert(result == CURAND_STATUS_SUCCESS);
}

__host__ void Distribution::callCuGetNormalSample(cuyasheint_t *array, int N, float mean, float stddev){
	curandGenerateLogNormal( gen, 
							(float*)array, 
							N,
							mean,
							stddev);

}