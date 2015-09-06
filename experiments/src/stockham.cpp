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
	complex* result;
    struct timespec start, stop;

	for(int N = FIRSTITERATION; N <= LASTITERATION; N *= 2){
		// const int N = 16;
		data = (complex*)malloc(N*sizeof(complex));
		data0 = (complex*)malloc(N*sizeof(complex));
		data1 = (complex*)malloc(N*sizeof(complex));
		result = (complex*)malloc(N*sizeof(complex));

		for(int i = 0; i < N; i++){
			if(i < N/2)
				data[i] = {rand()%1024,0};
			else
				data[i] = {0,0};
			data0[i] = data1[i] = {0,0};
		}

		for(int i = 0; i < N; i++)
			data0[i] = data[i];

		CPU_FFT(N,RADIX,data0,data1,FORWARD);
	    std::cout << "data 0 " << std::endl;
		for(int i=0;i < N;i++){
			std::cout << N << ") " << i << " - " << (data0[i].real) << std::endl;
			// assert(cu_result[i]/2 == data[i]);
	    }
	    std::cout << "data 1 " << std::endl;
	    for(int i=0;i < N;i++){
			std::cout << N << ") " << i << " - " << (data1[i].real) << std::endl;
			// assert(cu_result[i]/2 == data[i]);
	    }

		CPU_FFT(N,RADIX,data1,result,FORWARD);
	    std::cout << "data 1 " << std::endl;
		for(int i=0;i < N;i++){
			std::cout << N << ") " << i << " - " << (data1[i].real) << std::endl;
			// assert(cu_result[i]/2 == data[i]);
	    }
	    std::cout << "result " << std::endl;
	    for(int i=0;i < N;i++){
			std::cout << N << ") " << i << " - " << (result[i].real) << std::endl;
			// assert(cu_result[i]/2 == data[i]);
	    }
  //     	clock_gettime( CLOCK_REALTIME, &start);
		// for(int count = 0; count < NITERATIONS; count++)	
		// 	CPU_FFT(N,RADIX,data0,data1,FORWARD);
		// clock_gettime( CLOCK_REALTIME, &stop);
		// std::cout << N <<") CPU FT " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		
		// for(int count = 0; count < NITERATIONS; count++)	
		// 	CPU_FFT(N,RADIX,data0,data1,INVERSE);
		// clock_gettime( CLOCK_REALTIME, &stop);
		// std::cout << N <<") CPU IFFT) " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		
		
		// complex *cpu_result;
		// cpu_result = (complex*)malloc(N*sizeof(complex));
		// for(int i = 0; i < N; i++){
		// 	data0[i] = data[i];
		// 	data1[i] = {0,0};
		// 	cpu_result[i] = {0,0};
		// }
		// CPU_FFT(N,RADIX,data0,data1,FORWARD);
		// for(int i=0;i < N;i++){
		// 	std::cout << N << "A) " << i << " - " << (data0[i].real) << " == " << data[i].real << std::endl;
		// 	assert(data0[i].real == data[i].real);
	 //    }

		// CPU_FFT(N,RADIX,data1,cpu_result,INVERSE);
		// for(int i=0;i < N;i++){
		// 	std::cout << N << ") " << i << " - " << (cpu_result[i].real) << " == " << data[i].real << std::endl;
		// 	assert(cpu_result[i].real/N == data[i].real);
	 //    }

		// // const int N = 16;
		// cudaMalloc((void**)&d_data0,N*sizeof(float2));
		// cudaMalloc((void**)&d_data1,N*sizeof(float2));

		// h_data0 = (float2*)malloc(N*sizeof(float2));
		// h_data1 = (float2*)malloc(N*sizeof(float2));

		// for(int i = 0; i < N; i++)
		// 	h_data0[i] = {data[i].real,data[i].imag};
		// cudaMemcpy(d_data0,h_data0,N*sizeof(float2),cudaMemcpyHostToDevice);
		// cudaMemcpy(d_data1,h_data1,N*sizeof(float2),cudaMemcpyHostToDevice);
		// cudaError_t result;
  //     	clock_gettime( CLOCK_REALTIME, &start);
		// for(int count = 0; count < NITERATIONS; count++){	
		// 	CALL_GPU_FFT(N,RADIX,d_data0,d_data1,FORWARD);
		// 	result = cudaDeviceSynchronize();
		// 	assert(result == cudaSuccess);
		// }
		// clock_gettime( CLOCK_REALTIME, &stop);
		// std::cout << N <<") GPU FFT " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		
		// for(int count = 0; count < NITERATIONS; count++){	
		// 	CALL_GPU_FFT(N,RADIX,d_data0,d_data1,INVERSE);
		// 	result = cudaDeviceSynchronize();
		// 	assert(result == cudaSuccess);
		// }
		// clock_gettime( CLOCK_REALTIME, &stop);
		// std::cout << N <<") GPU IFFT) " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		
	

		// cudaMemcpy(d_data0,h_data0,N*sizeof(float2),cudaMemcpyHostToDevice);
		// cudaMemcpy(d_data1,h_data1,N*sizeof(float2),cudaMemcpyHostToDevice);
		// CALL_GPU_FFT(N,RADIX,d_data0,d_data1,FORWARD);
		// CALL_GPU_FFT(N,RADIX,d_data0,d_data1,INVERSE);

		// cu_result = (float2*)malloc(N*sizeof(float2));
		// cudaMemcpy(cu_result,d_data0,N*sizeof(float2),cudaMemcpyDeviceToHost);

		// for(int i=0;i < N;i++){
		// 	// std::cout << cu_result[i].x << std::endl; 
		// 	assert(abs( cu_result[i].x - data[i].real) < 0.01);
		// }

		// free(data0);
		// free(data1);
		// free(h_data0);
		// free(h_data1);
		// cudaFree(d_data0);
		// cudaFree(d_data1);
	}
	return 0;
}
