#include "stockham_reference.h"
// #include "stockham_global.h"

#include <cuda_runtime_api.h>
double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

int main(){	
	complex* data;
	complex* data0;
	complex* data1;
	float2* h_data0;
	float2* h_data1;
	float2* d_data0;
	float2* d_data1;
	float2* cu_result;
    struct timespec start, stop;

	for(int N = FIRSTITERATION; N <= LASTITERATION; N *= 2){
		// const int N = 16;
		data = (complex*)malloc(N*sizeof(complex));
		data0 = (complex*)malloc(N*sizeof(complex));
		data1 = (complex*)malloc(N*sizeof(complex));

		for(int i = 0; i < N; i++){
			if(i < N/2)
				data[i] = {rand()%1024,0};
			else
				data[i] = {0,0};
		}

		for(int i = 0; i < N; i++)
			data0[i] = data[i];
      	clock_gettime( CLOCK_REALTIME, &start);
		for(int count = 0; count < NITERATIONS; count++)	
			CPU_FFT(N,RADIX,data0,data1,FORWARD);
		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << N <<") CPU FT " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		
		for(int count = 0; count < NITERATIONS; count++)	
			CPU_FFT(N,RADIX,data0,data1,INVERSE);
		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << N <<") CPU IFFT) " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		

		// const int N = 16;
		cudaMalloc((void**)&d_data0,N*sizeof(float2));
		cudaMalloc((void**)&d_data1,N*sizeof(float2));

		h_data0 = (float2*)malloc(N*sizeof(float2));
		h_data1 = (float2*)malloc(N*sizeof(float2));

		for(int i = 0; i < N; i++)
			h_data0[i] = {data[i].real,data[i].imag};
		cudaMemcpy(d_data0,h_data0,N*sizeof(float2),cudaMemcpyHostToDevice);
		cudaMemcpy(d_data1,h_data1,N*sizeof(float2),cudaMemcpyHostToDevice);
		cudaError_t result;
      	clock_gettime( CLOCK_REALTIME, &start);
		for(int count = 0; count < NITERATIONS; count++){	
			CALL_GPU_FFT(N,RADIX,d_data0,d_data1,FORWARD);
			result = cudaDeviceSynchronize();
			assert(result == cudaSuccess);
		}
		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << N <<") GPU FFT " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		
		for(int count = 0; count < NITERATIONS; count++){	
			CALL_GPU_FFT(N,RADIX,d_data0,d_data1,INVERSE);
			result = cudaDeviceSynchronize();
			assert(result == cudaSuccess);
		}
		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << N <<") GPU IFFT) " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		
	

		cudaMemcpy(d_data0,h_data0,N*sizeof(float2),cudaMemcpyHostToDevice);
		cudaMemcpy(d_data1,h_data1,N*sizeof(float2),cudaMemcpyHostToDevice);
		CALL_GPU_FFT(N,RADIX,d_data0,d_data1,FORWARD);
		CALL_GPU_FFT(N,RADIX,d_data0,d_data1,INVERSE);

		cu_result = (float2*)malloc(N*sizeof(float2));
		cudaMemcpy(cu_result,d_data0,N*sizeof(float2),cudaMemcpyDeviceToHost);

		for(int i=0;i < N;i++){
			// std::cout << cu_result[i].x << std::endl; 
			assert(abs( cu_result[i].x - data[i].real) < 0.01);
		}

		free(data0);
		free(data1);
		free(h_data0);
		free(h_data1);
		cudaFree(d_data0);
		cudaFree(d_data1);
	}
	return 0;
}
