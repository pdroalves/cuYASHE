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

typedef float2 Complex;

__global__ void copyIntegerToComplex(Complex *a,uint32_t *b,int size){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size ){
      // a[tid].x =   __ull2double_rn(b[tid]);
      a[tid].x =   (b[tid]);
      // printf("%ld => %f\n\n",b[tid],a[tid].x);
      a[tid].y = 0;
  }else{
    a[tid].x = 0;
    a[tid].y = 0;
  }
}

__global__ void copyAndNormalizeComplexRealPartToInteger(uint32_t *b,const Complex *a,const int size,const double scale){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  uint32_t value;
  double fvalue;
  // double frac;
  if(tid < size ){
      fvalue = a[tid].x * scale;
      value = rint(fvalue);

      b[tid] = value;
  }
}

int check_errors(uint32_t *a, uint32_t *b, int size){
	int errors = 0;
	for(int i = 0; i < size;i++)
		if(a[i] != b[i])
			errors++;
	return errors;
}

void start_report(){
	std::cout << "from polynomial import Polynomial" << std::endl;
	std::cout << "import json" << std::endl;
	std::cout << "errors = {}" << std::endl;
	std::cout << "a = Polynomial()" << std::endl;
	std::cout << "retorno = Polynomial()" << std::endl;
    std::cout << "for CRT_primesize in range(9,63):" << std::endl;
    std::cout << "\terrors[CRT_primesize] = {}" << std::endl;
    std::cout << "\tfor i in range(1,14):" << std::endl;
    std::cout << "\t\terrors[CRT_primesize][2**i] = 0" << std::endl;
}

void finish_report(){
	std::cout << "print json.dumps(errors,indent=4)" <<std::endl;
}

void report(int SIZE,int prime_size, uint32_t *a, uint32_t *b){
	std::cout << "CRT_primesize = " << prime_size << std::endl;
	
	std::cout << "a.coef = [" << std::endl;
	for(int i = 0; i < SIZE; i++)
		std::cout << a[i] << ",";
	std::cout << "]"<< std::endl;

	std::cout << "b = a*a" << std::endl;

	std::cout << "retorno.coef = [" << std::endl;
	for(int i = 0; i < SIZE; i++)
		std::cout << b[i] << ",";
	std::cout << "]" << std::endl;	

    std::cout << "if b != retorno:" <<std::endl;
    std::cout << "\tfor i,coef in enumerate(b):" << std::endl;
    std::cout << "\t\tif i >= len(retorno.coef) or coef != retorno.coef[i]:" << std::endl;
    std::cout << "\t\t\terrors[CRT_primesize][" << SIZE << "] = errors[CRT_primesize][" << SIZE << "] + 1" << std::endl;

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

	start_report();

	int START_RANGE = atoi(argv[1]);
	int END_RANGE = atoi(argv[2]);
	int START_COEFFBITS = atoi(argv[3]);
	int END_COEFFBITS = atoi(argv[4]);
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
	uint32_t *h_input_array;
	uint32_t *d_input_array;
	uint32_t *h_output_array;
	uint32_t *d_output_array;
	Complex *d_complex_array;


	// Degrees
    for(int SIZE = START_RANGE; SIZE < END_RANGE;SIZE *= 2){
		
		fftResult = cufftPlan1d(&plan, SIZE, CUFFT_C2C, 1);
		assert(fftResult == CUFFT_SUCCESS);

		h_input_array = (uint32_t*)malloc(SIZE*sizeof(uint32_t));
		h_output_array = (uint32_t*)malloc(SIZE*sizeof(uint32_t));
		result = cudaMalloc((void**)&d_input_array,SIZE*sizeof(uint32_t));
		assert(result == cudaSuccess);
		result = cudaMalloc((void**)&d_output_array,SIZE*sizeof(uint32_t));
		assert(result == cudaSuccess);
		result = cudaMalloc((void**)&d_complex_array,SIZE*sizeof(Complex));
		assert(result == cudaSuccess);

	    result = cudaMemset(d_input_array,0,SIZE*sizeof(uint32_t));
	    assert(result == cudaSuccess);
	    result = cudaMemset(d_output_array,0,SIZE*sizeof(uint32_t));
	    assert(result == cudaSuccess);
	    result = cudaMemset(d_complex_array,0,SIZE*sizeof(Complex));
	    assert(result == cudaSuccess);
	    
	    // Primes
	    for(int COEFFBITS = START_COEFFBITS; COEFFBITS <= END_COEFFBITS; COEFFBITS++){

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
			result = cudaMemcpy(d_input_array,h_input_array,SIZE*sizeof(uint32_t),cudaMemcpyHostToDevice);
			assert(result == cudaSuccess);

			/** Convert to Complex */
			dim3 blockDim(32);
			dim3 gridDim((SIZE/32) + (SIZE%32));
			copyIntegerToComplex<<<gridDim,blockDim>>>(d_complex_array,d_input_array,SIZE);
			assert(cudaGetLastError() == cudaSuccess);

			/////////
			// FFT //
			/////////
		    fftResult = cufftExecC2C(plan, (cufftComplex *)(d_complex_array), (cufftComplex *)(d_complex_array), CUFFT_FORWARD);
		    assert(fftResult == CUFFT_SUCCESS);

		    //////////
		    // Mul  //
		    //////////
			mul<<< gridDim,blockDim>>>(d_complex_array, d_complex_array,d_complex_array,SIZE);

			//////////
			// IFFT //
			//////////
		    fftResult = cufftExecC2C(plan, (cufftComplex *)(d_complex_array), (cufftComplex *)(d_complex_array), CUFFT_INVERSE);
		    assert(fftResult == CUFFT_SUCCESS);

		    ////////////
		    // Scale  //
		    ////////////
		    copyAndNormalizeComplexRealPartToInteger<<< gridDim,blockDim>>>(d_output_array,d_complex_array,SIZE,1.0f/SIZE);
			assert(cudaGetLastError() == cudaSuccess);

			/** Copy */
			result = cudaMemcpy(h_output_array,d_output_array,SIZE*sizeof(uint32_t),cudaMemcpyDeviceToHost);
			assert(result == cudaSuccess);

			report(SIZE,COEFFBITS,h_input_array,h_output_array);
		}
	}
	finish_report();

	return 0;
}