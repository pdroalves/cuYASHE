#include "settings.h"
#include "cuda_bn.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>


__global__ void kernel(cuyasheint_t *value){
	printf("GPU: %d\n",value[0]);

	return;
}

void print(bn_t *value){
	std::cout << "CPU: " << value->dp[0] << std::endl;

	kernel<<<1,1>>>(value->dp);
}

int main(void){
	bn_t value;

	bn_new(&value);
	bn_zero(&value);

	print(&value);

	bn_set_dig(&value,42);

	print(&value);

	cudaDeviceSynchronize();

	return 0;
}