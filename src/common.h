	#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include "cuda_functions.h"
#include "settings.h"
#include <NTL/ZZ.h>
#include <NTL/ZZ_pEX.h>
#include <NTL/ZZ_pXFactoring.h>
#include <iomanip>
#include <time.h>

/////////////////////
// Auxiliar functions
uint64_t get_cycles();
double compute_time_ms(struct timespec start,struct timespec stop);
const std::string current_date_time() ;
int bitCount(unsigned int n);

bool check_overflow(uint64_t a,uint64_t b);

/////////////////////
// Common functions
template <class P>
P common_addition(P *a,P *b){
	// P should be Polynomial or Ciphertext
	#ifdef VERBOSE
	std::cout << "Operator+ on GPU" << std::endl;
	#endif
		// Check align
	int new_spacing = std::max(a->CRTSPACING,b->CRTSPACING);
	if(a->CRTSPACING != b->CRTSPACING){
	  a->update_crt_spacing(new_spacing);
	  b->update_crt_spacing(new_spacing);
	}

	#ifdef VERBOSE
	std::cout << "Add with CRTSPACING" << a->get_crt_spacing() << std::endl;
	// std::cout << "this: " << a->to_string() << std::endl;
	// std::cout << "other " << b->to_string() << std::endl;
	#endif

	// Apply CRT and copy data to global memory, if needed
	// #pragma omp parallel sections num_threads(2)
	{
	    // #pragma omp section
	    {
	      #ifdef VERBOSE
      	  std::cout << "a: " << std::endl;
      	  #endif
	      if(!a->get_transf_computed()){
	        a->update_device_data();
	      }

	    }
	    // #pragma omp section
	    {
	      #ifdef VERBOSE
       	  std::cout << "b: " << std::endl;
      	  #endif
       	  if(!b->get_transf_computed()){
	        b->update_device_data();
	      }
	    }
	}

	P c = P(a->get_mod(),a->get_phi(),new_spacing);
	CUDAFunctions::callPolynomialAddSub(c.get_device_crt_residues(),
										a->get_device_crt_residues(),
										b->get_device_crt_residues(),
										(int)(a->CRTSPACING*P::CRTPrimes.size()),
										ADD,
										a->get_stream());

	c.set_host_updated(false);
	c.set_icrt_computed(false);
	c.set_crt_computed(false);
	c.set_itransf_computed(false);
	c.set_transf_computed(true);

	return c;
}

template <class P>
void common_addition_inplace(P *a,P *b){
	// P should be Polynomial or Ciphertext
	//////////////////////////////////
	// Store result in polynomial a //
	//////////////////////////////////
	
	#ifdef VERBOSE
	std::cout << "Operator+= on GPU" << std::endl;
	#endif

	// Check align
	int new_spacing = std::max(a->CRTSPACING,b->CRTSPACING);
	if(a->CRTSPACING != b->CRTSPACING){
	  a->update_crt_spacing(new_spacing);
	  b->update_crt_spacing(new_spacing);
	}

	#ifdef VERBOSE
	std::cout << "Add with CRTSPACING" << a->get_crt_spacing() << std::endl;
	// std::cout << "this: " << a->to_string() << std::endl;
	// std::cout << "other " << b->to_string() << std::endl;
	#endif

	// Apply CRT and copy data to global memory, if needed
	// #pragma omp parallel sections num_threads(2)
	{
		// #pragma omp section
		{

			if(!a->get_transf_computed()){
				a->update_device_data();
			}

		}
		// #pragma omp section
		{
			if(!b->get_transf_computed()){
				b->update_device_data();
			}
		}
	}


	CUDAFunctions::callPolynomialAddSubInPlace( a->get_stream(),
												a->get_device_crt_residues(),
												b->get_device_crt_residues(),
												(int)(a->CRTSPACING*P::CRTPrimes.size()),
												ADD);

	a->set_host_updated(false);
	a->set_icrt_computed(false);
	a->set_crt_computed(false);
	a->set_itransf_computed(false);
	a->set_transf_computed(true);
}

template <class P>
P* common_multiplication(P *a, P *b){
	// P should be Polynomial or Ciphertext
	#ifdef VERBOSE
	std::cout << "Operator+ on GPU" << std::endl;
	#endif
		// Check align
	int new_spacing = std::max(a->get_crt_spacing(),b->get_crt_spacing());
	if(new_spacing < CUDAFunctions::N)
		new_spacing = CUDAFunctions::N;
	else if(new_spacing > CUDAFunctions::N)
		CUDAFunctions::init(new_spacing);
	new_spacing = CUDAFunctions::N;

	if(a->get_crt_spacing() != new_spacing)
		a->update_crt_spacing(new_spacing);
	if(b->get_crt_spacing() != new_spacing)
		b->update_crt_spacing(new_spacing);

	#ifdef VERBOSE
	std::cout << "Add with CRTSPACING" << a->get_crt_spacing() << std::endl;
	// std::cout << "this: " << a->to_string() << std::endl;
	// std::cout << "other " << b->to_string() << std::endl;
	#endif

	// Apply CRT and copy data to global memory, if needed
	// #pragma omp parallel sections num_threads(2)
	{
	    // #pragma omp section
	    {
	      #ifdef VERBOSE
      	  std::cout << "a: " << std::endl;
      	  #endif
	      if(!a->get_transf_computed()){
	        a->update_device_data();
	      }

	    }
	    // #pragma omp section
	    {
	      #ifdef VERBOSE
       	  std::cout << "b: " << std::endl;
      	  #endif
       	  if(!b->get_transf_computed()){
	        b->update_device_data();
	      }
	    }
	}

	P *c = new P(a->get_mod(),a->get_phi(),new_spacing);
	#ifdef NTTMUL_TRANSFORM
	CUDAFunctions::callPolynomialMul(  	c->get_device_crt_residues(),
										a->get_device_crt_residues(),
										b->get_device_crt_residues(),
										new_spacing*P::CRTPrimes.size(),
										a->get_stream());
	#else
	CUDAFunctions::executeCuFFTPolynomialMul( 	c->get_device_transf_residues(), 
	                                            a->get_device_transf_residues(), 
	                                            b->get_device_transf_residues(), 
	                                            new_spacing*P::CRTPrimes.size(),
	                                            b->get_stream());
	#endif

	c->set_host_updated(false);
	c->set_icrt_computed(false);
	c->set_crt_computed(false);
	c->set_itransf_computed(false);
	c->set_transf_computed(true);

	return c;
}

template <class P>
void common_multiplication_inplace(P *a, P *b){
	// P should be Polynomial or Ciphertext
	//////////////////////////////////
	// Store result in polynomial a //
	//////////////////////////////////
	
	#ifdef VERBOSE
	std::cout << "Operator+= on GPU" << std::endl;
	#endif

	int new_spacing = std::max(a->get_crt_spacing(),b->get_crt_spacing());
	// Check align
	if(new_spacing < CUDAFunctions::N)
		new_spacing = CUDAFunctions::N;
	else if(new_spacing > CUDAFunctions::N)
		CUDAFunctions::init(new_spacing);
	new_spacing = CUDAFunctions::N;

	if(a->get_crt_spacing() != new_spacing)
		a->update_crt_spacing(new_spacing);
	if(b->get_crt_spacing() != new_spacing)
		b->update_crt_spacing(new_spacing);

	#ifdef VERBOSE
	std::cout << "Add with CRTSPACING" << a->get_crt_spacing() << std::endl;
	// std::cout << "this: " << a->to_string() << std::endl;
	// std::cout << "other " << b->to_string() << std::endl;
	#endif

	// Apply CRT and copy data to global memory, if needed
	// #pragma omp parallel sections num_threads(2)
	{
		// #pragma omp section
		{

			if(!a->get_transf_computed()){
				a->update_device_data();
			}

		}
		// #pragma omp section
		{
			if(!b->get_transf_computed()){
				b->update_device_data();
			}
		}
	}


	CUDAFunctions::callPolynomialMul(   a->get_device_crt_residues(),
										a->get_device_crt_residues(),
										b->get_device_crt_residues(),
										new_spacing*P::CRTPrimes.size(),
										a->get_stream());

	a->set_host_updated(false);
	a->set_icrt_computed(false);
	a->set_crt_computed(false);
	a->set_itransf_computed(false);
	a->set_transf_computed(true);
}
#endif
