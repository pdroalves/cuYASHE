#include "stockham_reference.h"

double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

int main(int argc,char* argv[]){	
	complex* data;
	complex* data0;
	complex* data1;
	complex* result;
    struct timespec start, stop;

    assert(argc>1);

	for(int N = atoi(argv[1]); N <= atoi(argv[2]); N *= 2){
		// const int N = 16;
		data = (complex*)malloc(N*sizeof(complex));
		data0 = (complex*)malloc(N*sizeof(complex));
		data1 = (complex*)malloc(N*sizeof(complex));
		result = (complex*)malloc(N*sizeof(complex));

		// Gera dados aleatorios
		for(int i = 0; i < N; i++){
			if(i < N/2)
				data[i] = {rand()%1024,0};
			else
				data[i] = {0,0};
			data0[i] = data1[i] = {0,0};
		}

		// Copia
		for(int i = 0; i < N; i++)
			data0[i] = data[i];

		// Forward
		CPU_FFT(N,RADIX,data0,data1,FORWARD);
		CPU_FFT(N,RADIX,data1,result,INVERSE);
	    for(int i=0;i < N;i++){
			//assert(result[i] == data[i]);
	    }

 		clock_gettime( CLOCK_REALTIME, &start);
		for(int count = 0; count < NITERATIONS; count++)	
			CPU_FFT(N,RADIX,data0,data1,FORWARD);
		clock_gettime( CLOCK_REALTIME, &stop);
		
		std::cout << N <<" " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		
		
		free(data0);
		free(data1);
		free(result);
	}
	return 0;
}
