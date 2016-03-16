#include "settings.h"
#include "common.h"
#include "cuda_functions.h"
#include "polynomial.h"
#include <time.h>
#include <unistd.h>
#include <cstdlib>
// 
// This benchmark script aims to measure the latency of polynomial multiplication
// through NTT and cuFFT.
// 
//  We do not really care with cuYASHE's good practices or whatever. 
//   

#define BILLION  1000000000L
#define MILLION  1000000L
#define NITERATIONS 100   
/**
 * Compare all coefficients of two polynomials and returns how many are different
 * @param  FFT [description]
 * @param  NTT [description]
 * @return     [description]
 */
int cmp_coeffs(Polynomial FFT, Polynomial NTT){
  FFT.normalize();
  NTT.normalize();

  assert(FFT.deg() == NTT.deg());

  int diff = 0;
  for(int i = 0; i < FFT.deg(); i++)
    if(FFT.get_coeff(i) != NTT.get_coeff(i))
      diff++;
  return diff;
}
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]){

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

    // std::cout << "Starting!" <<std::endl;
  int CRT_primesize = atoi(argv[1]);
    std::cout << "#!/usr/bin/python" << std::endl;
    std::cout << "from polynomial import Polynomial" << std::endl;
    std::cout << "a = Polynomial()" << std::endl;
    std::cout << "b = Polynomial()" << std::endl;
    std::cout << "output_fft = Polynomial()" << std::endl;
    std::cout << "errors = {}" << std::endl;
    std::cout << "CRT_primesize = " << CRT_primesize << std::endl;

    std::cout << "errors[CRT_primesize] = {}" << std::endl;
    std::cout << "for i in range(1,14):" << std::endl;
    std::cout << "\terrors[CRT_primesize][2**i] = 0" << std::endl;

    for(int N = 512; N <= 8192; N *= 2){
      // sleep(3);
    	Polynomial::BuildNthCyclotomic(&phi, N); // generate an cyclotomic polynomial
      Polynomial::global_phi = &phi;
      // Polynomial::gen_crt_primes(Polynomial::global_mod,N);
      Polynomial::gen_crt_primes(Polynomial::global_mod,N,CRT_primesize);
  		CUDAFunctions::init(2*N);
  	
      for(int iterate = 0; iterate < 1; iterate++){
        /**
         * Setup
         */
        // std::cout << "Generating " << phi.deg() << " degree polynomials." << std::endl;
    		/** 
    		 * Variables and operands
    		 */
    		Polynomial a;
        Polynomial::random(&a,N-1);
        
        Polynomial b;
        Polynomial::random(&b,N-1);

        std::cout << "a.coef = [" << a.to_string() << "]" << std::endl;
        std::cout << "b.coef = [" << b.to_string() << "]" <<std::endl;
        /** 
         * Result cmp
         */
        
        /////////
        // NTT //
        /////////
        // CUDAFunctions::transform = NTTMUL;

        // Polynomial output_ntt = a*b;

        /////////
        // FFT //
        /////////
        // CUDAFunctions::init(2*N);
        CUDAFunctions::transform = CUFFTMUL;

        Polynomial output_fft = a*b;
        std::cout << "c = a*b" <<std::endl;
        std::cout << "output_fft.coef = [" << output_fft.to_string() << "]" <<std::endl;

        // std::cout << "output_ntt: " << output_ntt.to_string() << std::endl;
        // std::cout << "output_fft: " << output_fft.to_string() << std::endl;
        // std::cout << cmp_coeffs(output_fft,output_ntt) << " different coefficients" << std::endl;
        std::cout << "if c != output_fft:" <<std::endl;
        std::cout << "\tfor i,coef in enumerate(c):" << std::endl;
        std::cout << "\t\tif coef != output_fft.coef[i]:" << std::endl;
        std::cout << "\t\t\terrors[CRT_primesize][" << N << "] = errors[CRT_primesize][" << N << "] + 1" << std::endl;
        std::cout << std::endl;
      }
  	}

    std::cout << "import json" << std::endl << "print json.dumps(errors,indent=4)" << std::endl;

	return 0;
}