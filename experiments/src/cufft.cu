#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cufft.h>
#include <unistd.h>
#include <stdlib.h>

#define PRIMES_BUCKET
#define COEFFBITS 19

cufftHandle plan;

typedef double2 Complex;
 typedef unsigned long int uint64_t;

#define BILLION  1000000000L
#define MILLION  1000000L
#define NITERATIONS 100

__host__ double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

__host__ int bitCount(uint64_t n) {

    int counter = 0;
    while(n) {
        counter += n % 2;
        n >>= 1;
    }
    return counter;
 }

__global__ void copyIntegerToComplex(Complex *a,uint64_t *b,int size){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size ){
      a[tid].x =   __ull2double_rn(b[tid]);
      // printf("%ld => %f\n\n",b[tid],a[tid].x);
      a[tid].y = 0;
  }else{
    a[tid].x = 0;
    a[tid].y = 0;
  }
}

__global__ void copyAndNormalizeComplexRealPartToInteger(uint64_t *b,const Complex *a,const int size,const double scale){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  uint64_t value;
  double fvalue;
  // double frac;
  if(tid < size ){
      fvalue = a[tid].x * scale;
      value = rint(fvalue);

      b[tid] = value;
  }
}

__device__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}


// Complex pointwise multiplication
__global__ void mul(const Complex *a, const Complex *b,Complex *c,int size){
    const int tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid < size  ){
        c[tid] = ComplexMul(a[tid], b[tid]);
    }else{
      c[tid].x = 0;
      c[tid].y = 0;
    }
}
/**
 * Main function
 * @param  argc [description]
 * @param  argv 1: Number of coefficients. 2-3: Interval of #Bits of each coefficient.
 * @return      [description]
 */
int main(int argc, char* argv[]){

	assert(argc >= 3);

	int START_RANGE = atoi(argv[1]);
	int END_RANGE = atoi(argv[2]);

	/////////////////
	// Setup cuFFT //
	/////////////////
	cufftResult fftResult;

	// std::cout << "Plan created with signal size " << SIZE << std::endl;

	///////////////////////////
	// Generate polynomials. //
	///////////////////////////
	cudaError_t result;

	/** Memory alloc */
	uint64_t *h_input_array;
	uint64_t *d_input_array;
	Complex *d_complex_array;

    struct timespec start, stop;

	// Degrees
    for(int SIZE = 2*START_RANGE; SIZE <= 2*END_RANGE;SIZE *= 2){
		// Polynomial multiplication of two operands with degree SIZE/2
		
		fftResult = cufftPlan1d(&plan, SIZE, CUFFT_Z2Z, 1);
		assert(fftResult == CUFFT_SUCCESS);

		h_input_array = (uint64_t*)malloc(SIZE*sizeof(uint64_t));
		result = cudaMalloc((void**)&d_input_array,SIZE*sizeof(uint64_t));
		assert(result == cudaSuccess);
		result = cudaMalloc((void**)&d_complex_array,SIZE*sizeof(Complex));
		assert(result == cudaSuccess);

	    result = cudaMemset(d_input_array,0,SIZE*sizeof(uint64_t));
	    assert(result == cudaSuccess);
	    result = cudaMemset(d_complex_array,0,SIZE*sizeof(Complex));
	    assert(result == cudaSuccess);
	    
		/** Generate random coeficients mod COEFFBITS*/
		for(int j = 0; j < SIZE; j++)
			if(j < SIZE/2){
				int min = ((uint64_t)2<<(COEFFBITS-2));
				int max = ((uint64_t)2<<COEFFBITS-1); 	

				int random = (rand()%(max-min));
				h_input_array[j] = min + random;
			}
			else
				h_input_array[j] = 0;

		/** Copy */
		result = cudaMemcpy(d_input_array,h_input_array,SIZE*sizeof(uint64_t),cudaMemcpyHostToDevice);
		assert(result == cudaSuccess);

		/** Convert to Complex */
		dim3 blockDim(32);
		dim3 gridDim((SIZE/32) + (SIZE%32));
		copyIntegerToComplex<<<gridDim,blockDim>>>(d_complex_array,d_input_array,SIZE);
		assert(cudaGetLastError() == cudaSuccess);

		/////////
		// FFT //
		/////////
 		clock_gettime( CLOCK_REALTIME, &start);
		for(int count = 0; count < NITERATIONS; count++)	
		    fftResult = cufftExecZ2Z(plan, (cufftDoubleComplex *)(d_complex_array), (cufftDoubleComplex *)(d_complex_array), CUFFT_FORWARD);
	    assert(fftResult == CUFFT_SUCCESS);
 		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << (SIZE/2) <<" " << compute_time_ms(start,stop)/NITERATIONS << " ms" << std::endl<< std::endl;		

		free(h_input_array);
		cudaFree(d_input_array);
		cudaFree(d_complex_array);
	}

	return 0;
}
