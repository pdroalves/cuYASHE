#include "settings.h"
#include "common.h"
#include "cuda_functions.h"
#include "polynomial.h"
#include <time.h>
#include <unistd.h>

// 
// This benchmark script aims to measure the latency of polynomial multiplication
// through NTT and cuFFT.
// 
//  We do not really care with cuYASHE's good practices or whatever. 
//   

#define BILLION  1000000000L
#define MILLION  1000000L
#define NITERATIONS 100   

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/**
 * We do not want to edit cuYASHE's flags and recompile it everytime.
 * So, we need to re-implement part of cuFFT setup here.
 */

typedef double2 Complex;

void fft_setup(int N){
    cufftResult fftResult;

    // # of CRT residues
    const int batch = Polynomial::CRTPrimes.size();
    assert(batch > 0);

    // # 1 dimensional FFT
    const int rank = 1;

    // No idea what is this
    int n[1] = {N};

    fftResult = cufftPlanMany(&CUDAFunctions::plan, rank, n,
         NULL, 1, N,  //advanced data layout, NULL shuts it off
         NULL, 1, N,  //advanced data layout, NULL shuts it off
         CUFFT_Z2Z, batch);
    // fftResult = cufftPlan1d(&CUDAFunctions::plan, N, CUFFT_Z2Z, 1);


    assert(fftResult == CUFFT_SUCCESS);
    std::cout << "Plan created with signal size " << N << std::endl;
}

void fft(	cuyasheint_t *output,
	        cuyasheint_t *a,
	        cuyasheint_t *b,
	        const int N,
	        const int NPolis,
	        cudaStream_t stream
        ){

  //////////////////////
  // cuFFT Setup init //
  //////////////////////
  const int size = N*NPolis;
  const int size_c = N;
  const int signal_size = N;
  cudaError_t result;
  Complex *d_a = CUDAFunctions::d_mulComplexA;
  Complex *d_b = CUDAFunctions::d_mulComplexB;
  Complex *d_c = CUDAFunctions::d_mulComplexC;

  result = cudaMemsetAsync(d_a,0,size*sizeof(Complex),stream);
  assert(result == cudaSuccess);
  result = cudaMemsetAsync(d_b,0,size*sizeof(Complex),stream);
  assert(result == cudaSuccess);
  result = cudaMemsetAsync(d_c,0,size*sizeof(Complex),stream);
  assert(result == cudaSuccess);

  dim3 blockDim(32);
  dim3 gridDim(size/32 + (size % 32 == 0? 0:1));

  CUDAFunctions::executeCopyIntegerToComplex(d_a,a,size,stream);
  assert(cudaGetLastError() == cudaSuccess);
  CUDAFunctions::executeCopyIntegerToComplex(d_b,b,size,stream);
  assert(cudaGetLastError() == cudaSuccess);

  /**
   * Stream set
   */
  cufftResult fftResult;
  fftResult = cufftSetStream(CUDAFunctions::plan, stream);
  assert(fftResult == CUFFT_SUCCESS);

  /////////////////////
  // cuFFT Setup end //
  /////////////////////
  ///
  /**
   * FFT
   */
  fftResult = cufftExecZ2Z(CUDAFunctions::plan, (cufftDoubleComplex *)(d_a), (cufftDoubleComplex *)(d_a), CUFFT_FORWARD);
  assert(fftResult == CUFFT_SUCCESS);

  fftResult = cufftExecZ2Z(CUDAFunctions::plan, (cufftDoubleComplex *)(d_b), (cufftDoubleComplex *)(d_b), CUFFT_FORWARD);
  assert(fftResult == CUFFT_SUCCESS);

  /**
   * MUL
   */
  CUDAFunctions::executeCuFFTPolynomialMul(d_a,d_b,d_c,size_c,size,stream);
  assert(cudaGetLastError() == cudaSuccess);

  /**
   * IFFT
   */
  fftResult = cufftExecZ2Z(CUDAFunctions::plan, (cufftDoubleComplex *)(d_c), (cufftDoubleComplex *)(d_c), CUFFT_INVERSE);
  assert(fftResult == CUFFT_SUCCESS);

  /**
   * Normalize
   */
  CUDAFunctions::executeCopyAndNormalizeComplexRealPartToInteger(output,(cufftDoubleComplex *)d_c,size,1.0f/signal_size,N,stream);
  assert(cudaGetLastError() == cudaSuccess);
  result = cudaDeviceSynchronize();

}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/////////
// NTT //
/////////
void ntt(	cuyasheint_t *output,
	        cuyasheint_t *a,
	        cuyasheint_t *b,
	        const int N,
	        const int NPolis,
	        cudaStream_t stream
        ){

  cuyasheint_t *d_result = output;

  const int size = N*NPolis;
  cuyasheint_t *d_a = CUDAFunctions::d_mulA;
  cuyasheint_t *d_b = CUDAFunctions::d_mulB;
  cuyasheint_t *aux = CUDAFunctions::d_mulAux;

  cudaError_t result;

  result = cudaMemsetAsync(aux,0,size*sizeof(cuyasheint_t),stream);
  assert(result == cudaSuccess);
  result = cudaMemsetAsync(d_result,0,size*sizeof(cuyasheint_t),stream);
  assert(result == cudaSuccess);
  // result = cudaMemsetAsync(mem,0,(4*size)*sizeof(cuyasheint_t),stream);
  // assert(result == cudaSuccess);

  // result = cudaMemcpyAsync(d_a,a,size*sizeof(cuyasheint_t),cudaMemcpyDeviceToDevice,stream);
  // assert(result == cudaSuccess);
  // result = cudaMemcpyAsync(d_b,b,size*sizeof(cuyasheint_t),cudaMemcpyDeviceToDevice,stream);
  // assert(result == cudaSuccess);

  int RADIX;
  /*if(N % 8 == 0)
    RADIX = 8;
  else*/ if(N % 4 == 0)
    RADIX = 4;
  else{
    assert(N % 2 == 0);
    RADIX = 2;
  }
  // const int RADIX = 2;
  dim3 blockDim(std::min(N/RADIX,1024));
  dim3 gridDim(NPolis);

  // Forward
  d_a = CUDAFunctions::applyNTT(d_a, N, NPolis, FORWARD, stream);

  result = cudaMemsetAsync(aux,0,size*sizeof(cuyasheint_t),stream);
  assert(result == cudaSuccess);

  // Inverse
  d_b = CUDAFunctions::applyNTT(d_b, N, NPolis, FORWARD, stream);

  result = cudaMemsetAsync(aux,0,size*sizeof(cuyasheint_t),stream);
  assert(result == cudaSuccess);

  // Multiply
  dim3 blockDimMul(ADDBLOCKXDIM);
  dim3 gridDimMul((size)/ADDBLOCKXDIM+1); // We expect that ADDBLOCKXDIM always divide size
  CUDAFunctions::executePolynomialMul(d_a,d_a,d_b,size,stream);
  assert(cudaGetLastError() == cudaSuccess);

  // Inverse
  d_a = CUDAFunctions::applyNTT(d_a, N, NPolis, INVERSE, stream);

  std::swap(d_a,d_result);

  CUDAFunctions::executeNTTScale(d_result,size,N, stream);
  
  // cudaFree(d_a);
  // cudaFree(d_b);
  // cudaFree(aux);
  result = cudaDeviceSynchronize();
}
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(void){

	struct timespec start, stop;
	float diff;
	clock_gettime( CLOCK_REALTIME, &start);
	sleep(1);
	clock_gettime( CLOCK_REALTIME, &stop);
	diff = compute_time_ms(start,stop);
	std::cout << "1 sec: " << diff << std::endl;

	Polynomial phi;
	ZZ q;

    NTL::power2(q,127);
    q -= conv<ZZ>("1");
	ZZ_p::init(q); // Defines GF(q)
	Polynomial::global_mod = q; // Defines default GF(q)

    std::cout << "Starting!" <<std::endl;

	for(int N = 1024; N <= 8192; N *= 2){
		/**
		 * Setup
		 */
    	Polynomial::BuildNthCyclotomic(&phi, N); // generate an cyclotomic polynomial
	    Polynomial::gen_crt_primes(Polynomial::global_mod,N);
		CUDAFunctions::init(2*N);
	
	    std::cout << "Generating " << phi.deg() << " degree polynomials." << std::endl;
    	std::cout << "O CRT usa " << Polynomial::CRTPrimes.size() << " primos de " << CRTPRIMESIZE << " bits"<< std::endl;

		/** 
		 * Variables and operands
		 */
		Polynomial a;
	    Polynomial::random(&a,N);
	    a.update_crt_spacing(2*N);
    	a.update_device_data();
		
		Polynomial b;
	    Polynomial::random(&b,N);
	    b.update_crt_spacing(2*N);
	    b.update_device_data();

		Polynomial output;
	    Polynomial::random(&output,N); // This isn't really needed
	    output.update_crt_spacing(2*N);
	    output.update_device_data();

	    /**
	     * Benchmark
	     */
	    /////////
	    // NTT //
	    /////////
	    uint64_t clock_start,clock_stop;
    	clock_gettime( CLOCK_REALTIME, &start);
    	clock_start = get_cycles();
	    for(int i = 0; i < NITERATIONS; i++)
			ntt( output.get_device_crt_residues(),
				a.get_device_crt_residues(), 
				b.get_device_crt_residues(),
				N,
				Polynomial::CRTPrimes.size(),
				0x0);
    	clock_gettime( CLOCK_REALTIME, &stop);
	    diff = compute_time_ms(start,stop);
	    std::cout << "NTT: " << diff/NITERATIONS << " ms." << std::endl;
		
		/////////
		// FFT //
		/////////
	    Polynomial::gen_crt_primes(Polynomial::global_mod,N,9);
		CUDAFunctions::init(2*N);
		fft_setup(2*N);
    	std::cout << "O CRT usa " << Polynomial::CRTPrimes.size() << " primos de " << 9 << " bits"<< std::endl;
		

	    Polynomial::random(&a,N);
	    a.update_crt_spacing(2*N);
    	a.update_device_data();
	    Polynomial::random(&b,N);
	    b.update_crt_spacing(2*N);
	    b.update_device_data();
	    Polynomial::random(&output,N); // This isn't really needed
	    output.update_crt_spacing(2*N);
	    output.update_device_data();

		clock_gettime( CLOCK_REALTIME, &start);
	    for(int i = 0; i < NITERATIONS; i++)
			fft( output.get_device_crt_residues(),
				a.get_device_crt_residues(), 
				b.get_device_crt_residues(),
				N,
				Polynomial::CRTPrimes.size(),
				0x0);
    	clock_gettime( CLOCK_REALTIME, &stop);
	    diff = compute_time_ms(start,stop);
	    std::cout << "FFT: " << diff/NITERATIONS << " ms." << std::endl;
		std::cout << std::endl;
	}


	return 0;
}