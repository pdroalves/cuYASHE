#include "stockham_ntt_reference.h"
#include <NTL/ZZ.h>
#include <fstream>
NTL_CLIENT

double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

int main(int argc,char* argv[]){	
	integer* data;
	integer* data0;
	integer* data1;
	integer *result;
	integer *h_W;
	integer *h_WInv;

    struct timespec start, stop;

	for(int N = atoi(argv[1]); N <= atoi(argv[2]); N *= 2){
		// wN computation
		ZZ PZZ = conv<ZZ>("18446744069414584321");
		integer k = conv<integer>(PZZ-1)/N;
		assert( (PZZ-1) % N == 0);
		ZZ wNZZ = NTL::PowerMod(ZZ(7),k,PZZ);
		integer wN = conv<integer>(wNZZ);
		assert(NTL::PowerMod(wNZZ,N,PZZ) == 1);

		h_W = (integer*)malloc(N*sizeof(integer));
		h_WInv = (integer*)malloc(N*sizeof(integer));

		// Computes 1-th column from W
		for(int j = 0; j < N; j++)
		    h_W[j] = conv<integer>(NTL::PowerMod(wNZZ,j,PZZ));

		// Computes 1-th column from WInv
		for(int j = 0; j < N; j++)
		      h_WInv[j] = conv<integer>(NTL::InvMod(conv<ZZ>(h_W[j]),PZZ ));

		//////////////////////
		// const int N = 16;
		data = (integer*)malloc(N*sizeof(integer));
		data0 = (integer*)malloc(N*sizeof(integer));
		data1 = (integer*)malloc(N*sizeof(integer));
		result = (integer*)malloc(N*sizeof(integer));

		// Raw data
		for(int i = 0; i < N; i++)
			if(i < N/2)
				data[i] = rand()%1024;
			else
				data[i] = 0;

		// Copy data
		for(int i = 0; i < N; i++)
			data0[i] = data[i];

		CPU_NTT(h_W,h_WInv,N,RADIX,data0,data1,FORWARD);
		CPU_NTT(h_W,h_WInv,N,RADIX,data1,result,INVERSE);
		for(int i=0;i < N;i++)
			assert(data[i] == result[i]/2);
	    
 		clock_gettime( CLOCK_REALTIME, &start);
		for(int count = 0; count < NITERATIONS; count++)	
			CPU_NTT(h_W,h_WInv,N,RADIX,data0,data1,FORWARD);
		clock_gettime( CLOCK_REALTIME, &stop);
		
		std::cout << N <<") CPU NTT " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		
		
		free(data0);
		free(data1);
		free(result);

	}
	return 0;
}
