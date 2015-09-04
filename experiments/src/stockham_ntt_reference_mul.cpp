#include "stockham_ntt_reference.h"
#include <NTL/ZZ.h>
#include <fstream>
NTL_CLIENT

#include <cuda_runtime_api.h>
double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

int main(){
	ofstream cpu_ntt;
	ofstream cpu_intt;
	ofstream gpu_ntt;
	ofstream gpu_intt;

	cpu_ntt.open ("cpu_ntt.dat");
	cpu_intt.open ("cpu_intt.dat");
	gpu_ntt.open ("gpu_ntt.dat");
	gpu_intt.open ("gpu_intt.dat");

	integer* data;
	integer* data0;
	integer* data1;
	integer* h_data0;
	integer* h_data1;
	integer* d_data0;
	integer* d_data1;
	integer *d_W;
	integer *d_WInv;
	integer *h_W;
	integer *h_WInv;
	integer* cpu_result;
	integer* cu_result;
	integer* d_cu_result;

    struct timespec start, stop;

	for(int N = FIRSTITERATION; N <= LASTITERATION; N *= 2){
		// wN computation


		ZZ PZZ = conv<ZZ>("18446744069414584321");
		integer k = conv<integer>(PZZ-1)/N;
		assert( (PZZ-1) % N == 0);
		ZZ wNZZ = NTL::PowerMod(ZZ(7),k,PZZ);
		integer wN = conv<integer>(wNZZ);
		assert(NTL::PowerMod(wNZZ,N,PZZ) == 1);
		std::cout << NTL::PowerMod(ZZ(wN),N,PZZ);

		cudaError_t result;
		h_W = (integer*)malloc(N*sizeof(integer));
		result = cudaMalloc((void**)&d_W,N*sizeof(integer));
		assert(result == cudaSuccess);
		h_WInv = (integer*)malloc(N*sizeof(integer));
		result = cudaMalloc((void**)&d_WInv,N*sizeof(integer));
		assert(result == cudaSuccess);

		// Computes 1-th column from W
		for(int j = 0; j < N; j++)
		    h_W[j] = conv<integer>(NTL::PowerMod(wNZZ,j,PZZ));

		// Computes 1-th column from WInv
		for(int j = 0; j < N; j++)
		      h_WInv[j] = conv<integer>(NTL::InvMod(conv<ZZ>(h_W[j]),PZZ ));

		result = cudaMemcpy (d_W,h_W , N*sizeof(integer),cudaMemcpyHostToDevice);
		assert(result == cudaSuccess);
		result = cudaMemcpy(d_WInv,h_WInv , N*sizeof(integer),cudaMemcpyHostToDevice);
		assert(result == cudaSuccess);


		//////////////////////
		// const int N = 16;
		data = (integer*)malloc(N*sizeof(integer));
		data0 = (integer*)malloc(N*sizeof(integer));
		data1 = (integer*)malloc(N*sizeof(integer));
		cpu_result = (integer*)malloc(N*sizeof(integer));


		// Raw data
		for(int i = 0; i < N; i++)
			if(i < N/2)
				data[i] = rand()%1024;
			else
				data[i] = 0;

		// CPU-NTT data
		for(int i = 0; i < N; i++)
			data0[i] = data[i];

		// CPU-NTT Forward
	  	clock_gettime( CLOCK_REALTIME, &start);
		for(int count = 0; count < NITERATIONS; count++)
			CPU_NTT(h_W,h_WInv,N,RADIX,data0,data1,FORWARD);
		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << N <<") CPU NTT " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;
		cpu_ntt << N << " " << compute_time_ms(start,stop)/NITERATIONS << std::endl;

		// CPU-NTT Inverse
		for(int count = 0; count < NITERATIONS; count++)
			CPU_NTT(h_W,h_WInv,N,RADIX,data1,data0,INVERSE);
		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << N <<") CPU INTT) " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;
		cpu_intt << N << " " << compute_time_ms(start,stop)/NITERATIONS << std::endl;

		// Verify correctness
		for(int i = 0; i < N; i++)
			data0[i] = data[i];

		CPU_NTT(h_W,h_WInv,N,RADIX,data0,data1,FORWARD);
		CPU_NTT(h_W,h_WInv,N,RADIX,data1,cpu_result,INVERSE);
		for(int i=0;i < N;i++){
			std::cout << N << ") " << i << " - " << (cpu_result[i]) << " == " << data[i] << std::endl;
			// assert(cu_result[i]/2 == data[i]);
	    }

		// GPU
		cudaMalloc((void**)&d_data0,N*sizeof(integer));
		cudaMalloc((void**)&d_data1,N*sizeof(integer));

		h_data0 = (integer*)malloc(N*sizeof(integer));
		h_data1 = (integer*)malloc(N*sizeof(integer));

		// GPU-NTT data
		for(int i = 0; i < N; i++)
			h_data0[i] = data[i];
		cudaMemcpy(d_data0,h_data0,N*sizeof(integer),cudaMemcpyHostToDevice);
		cudaMemcpy(d_data1,h_data1,N*sizeof(integer),cudaMemcpyHostToDevice);

		// GPU-NTT Forward
	  	clock_gettime( CLOCK_REALTIME, &start);
		for(int count = 0; count < NITERATIONS; count++){
			CALL_GPU_NTT(d_W,d_WInv,N,RADIX,d_data0,d_data1,FORWARD);
			result = cudaDeviceSynchronize();
			assert(result == cudaSuccess);
		}
		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << N <<") GPU NTT " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;
		gpu_ntt << N << " " << compute_time_ms(start,stop)/NITERATIONS << std::endl;

		// GPU-NTT Inverse
		for(int count = 0; count < NITERATIONS; count++){
			CALL_GPU_NTT(d_W,d_WInv,N,RADIX,d_data1,d_data0,INVERSE);
			result = cudaDeviceSynchronize();
			assert(result == cudaSuccess);
		}
		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << N <<") GPU INTT) " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;
		gpu_intt << N << " " << compute_time_ms(start,stop)/NITERATIONS << std::endl;

		cu_result = (integer*)malloc(N*sizeof(integer));
		cudaMalloc((void**)&d_cu_result,N*sizeof(integer));

		cudaMemcpy(d_data0,h_data0,N*sizeof(integer),cudaMemcpyHostToDevice);
		cudaMemcpy(d_data1,h_data1,N*sizeof(integer),cudaMemcpyHostToDevice);
		CALL_GPU_NTT(d_W,d_WInv,N,RADIX,d_data0,d_data1,FORWARD);
		cudaMemcpy(cu_result,d_data1,N*sizeof(integer),cudaMemcpyDeviceToHost);
		CALL_GPU_NTT(d_W,d_WInv,N,RADIX,d_data1,d_cu_result,INVERSE);
		cudaMemcpy(cu_result,d_cu_result,N*sizeof(integer),cudaMemcpyDeviceToHost);

		// Correctness test
		for(int i=0;i < N;i++){
			std::cout << N << ") " << i << " - " << (cu_result[i]) << " == " << data[i] << std::endl;
			assert(cu_result[i]/2 == data[i]);
	    }
		free(data0);
		free(data1);
		free(h_data0);
		free(h_data1);
		free(h_W);
		free(h_WInv);
		cudaFree(d_W);
		cudaFree(d_WInv);
		cudaFree(d_data0);
		cudaFree(d_data1);
	}
	return 0;
}
