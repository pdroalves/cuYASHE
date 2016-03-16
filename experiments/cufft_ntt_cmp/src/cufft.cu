#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cufft.h>
#include <unistd.h>
#include <stdlib.h>

cufftHandle plan;

typedef double2 Complex;

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

int check_errors(uint64_t *a, uint64_t *b, int size){
	int errors = 0;

	for(int i = 0; i < size;i++){
		// std::cout << a[i] << " == " << b[i] << std::endl;
		if(a[i] != b[i])
			errors++;
	}
	return errors;
}

void start_report(){
	std::cout << "errors = {" << std::endl;
}

void finish_report(){
	std::cout << "}" << std::endl;
	std::cout << "import json" << std::endl;
	std::cout << "print json.dumps(errors,indent=4)" << std::endl;
}

void report(int SIZE,int COEFFBITS,int errors){
	std::cout << "\t" << COEFFBITS << ":" << errors << "," << std::endl;
}

/**
 * Main function
 * @param  argc [description]
 * @param  argv 1: Number of coefficients. 2-3: Interval of #Bits of each coefficient.
 * @return      [description]
 */
int main(int argc, char* argv[]){

	assert(argc >= 3);

	start_report();

	int SIZE = atoi(argv[1]);
	int START_COEFFBITS = atoi(argv[2]);
	int END_COEFFBITS = atoi(argv[3]);
	/////////////////
	// Setup cuFFT //
	/////////////////
	cufftResult fftResult;

	fftResult = cufftPlan1d(&plan, SIZE, CUFFT_Z2Z, 1);
	assert(fftResult == CUFFT_SUCCESS);
	// std::cout << "Plan created with signal size " << SIZE << std::endl;

	///////////////////////////
	// Generate polynomials. //
	///////////////////////////
	cudaError_t result;

	/** Memory alloc */
	uint64_t *h_input_array;
	uint64_t *d_input_array;
	uint64_t *h_output_array;
	uint64_t *d_output_array;
	Complex *d_complex_array;

	h_input_array = (uint64_t*)malloc(SIZE*sizeof(uint64_t));
	h_output_array = (uint64_t*)malloc(SIZE*sizeof(uint64_t));
	result = cudaMalloc((void**)&d_input_array,SIZE*sizeof(uint64_t));
	assert(result == cudaSuccess);
	result = cudaMalloc((void**)&d_output_array,SIZE*sizeof(uint64_t));
	assert(result == cudaSuccess);
	result = cudaMalloc((void**)&d_complex_array,SIZE*sizeof(Complex));
	assert(result == cudaSuccess);

    result = cudaMemset(d_input_array,0,SIZE*sizeof(uint64_t));
    assert(result == cudaSuccess);
    result = cudaMemset(d_output_array,0,SIZE*sizeof(uint64_t));
    assert(result == cudaSuccess);
    result = cudaMemset(d_complex_array,0,SIZE*sizeof(Complex));
    assert(result == cudaSuccess);

    for(int COEFFBITS = START_COEFFBITS; COEFFBITS <= END_COEFFBITS; COEFFBITS++){

		/** Generate random coeficients mod COEFFBITS*/
		for(int j = 0; j < SIZE; j++)
			h_input_array[j] = ( ((uint64_t)rand() << 32) + rand() ) % ((uint64_t)2<<COEFFBITS);
		/** Copy */
		result = cudaMemcpy(d_input_array,h_input_array,SIZE*sizeof(uint64_t),cudaMemcpyHostToDevice);
		assert(result == cudaSuccess);

		/** Convert to Complex */
		dim3 blockDim(32);
		dim3 gridDim((SIZE/32) + (SIZE%32));
		copyIntegerToComplex<<<gridDim,blockDim>>>(d_complex_array,d_input_array,SIZE);
		assert(cudaGetLastError() == cudaSuccess);

		// FFT -> IFFT -> FFT
	    fftResult = cufftExecZ2Z(plan, (cufftDoubleComplex *)(d_complex_array), (cufftDoubleComplex *)(d_complex_array), CUFFT_FORWARD);
	    assert(fftResult == CUFFT_SUCCESS);
	    fftResult = cufftExecZ2Z(plan, (cufftDoubleComplex *)(d_complex_array), (cufftDoubleComplex *)(d_complex_array), CUFFT_INVERSE);
	    assert(fftResult == CUFFT_SUCCESS);

	    copyAndNormalizeComplexRealPartToInteger<<< gridDim,blockDim>>>(d_output_array,d_complex_array,SIZE,1.0f/SIZE);
		assert(cudaGetLastError() == cudaSuccess);

		/** Copy */
		result = cudaMemcpy(h_output_array,d_output_array,SIZE*sizeof(uint64_t),cudaMemcpyDeviceToHost);
		assert(result == cudaSuccess);

		report(SIZE,COEFFBITS,check_errors(h_input_array,h_output_array,SIZE));
	}
	finish_report();

	return 0;
}