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

uint64_t get_cycles();
double compute_time_ms(struct timespec start,struct timespec stop);
const std::string current_date_time() ;
int bitCount(unsigned int n);

bool check_overflow(uint64_t a,uint64_t b);

template <class P>
P common_addition(P *a,P *b){
	// P should be Polynomial or Ciphertext
	#ifdef MAYADDONCPU
		if(!a->get_device_updated() && !b->get_device_updated()){
			// CPU add
			#ifdef VERBOSE
			std::cout << "Operator+ on CPU" << std::endl;
			#endif
			P c(*this);
			c.CPUAddition(&b);
			return c;
		}else{
	#endif
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
			      if(!a->get_device_updated()){
			        a->update_device_data();
			      }

			    }
			    // #pragma omp section
			    {
			      #ifdef VERBOSE
		       	  std::cout << "b: " << std::endl;
		      	  #endif
		       	  if(!b->get_device_updated()){
			        b->update_device_data();
			      }
			    }
			}

			cuyasheint_t *d_result = CUDAFunctions::callPolynomialAddSub(a->get_stream(),a->get_device_crt_residues(),b->get_device_crt_residues(),(int)(a->CRTSPACING*P::CRTPrimes.size()),ADD);

			P c = P(a->get_mod(),a->get_phi(),new_spacing);
			c.set_device_crt_residues(d_result);
			c.set_host_updated(false);
			c.set_device_updated(true);
			// cudaDeviceSynchronize();
			return c;
	#ifdef MAYADDONCPU
		}
	#endif
}

template <class P>
void common_addition_inplace(P *a,P *b){
	// P should be Polynomial or Ciphertext
	// Store result in polynomial a
	
	#ifdef VERBOSE
	std::cout << "Operator+= on GPU" << std::endl;
	#endif

	// Check align
	if(a->get_crt_spacing() != b->get_crt_spacing()){
		int new_spacing = std::max(a->get_crt_spacing(),b->get_crt_spacing());
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

			if(!a->get_device_updated()){
				a->update_device_data();
			}

		}
		// #pragma omp section
		{
			if(!b->get_device_updated()){
				b->update_device_data();
			}
		}
	}


	CUDAFunctions::callPolynomialAddSubInPlace( a->get_stream(),
												a->get_device_crt_residues(),
												b->get_device_crt_residues(),
												(int)(a->get_crt_spacing()*P::CRTPrimes.size()),
												ADD);

	a->set_host_updated(false);
	a->set_device_updated(true);
}

template <class P>
P common_multiplication(P *a_input, P *b_input){
  
  uint64_t start,end;

  P a,b;
  a.copy(a_input);
  b.copy(b_input);
  // Check align
  int needed_spacing = pow(2,ceil(log2(a.get_crt_spacing()+b.get_crt_spacing())));
  
  if(needed_spacing < CUDAFunctions::N)
	needed_spacing = CUDAFunctions::N;
  else if(needed_spacing != CUDAFunctions::N)
	// Re-compute W matrix
	CUDAFunctions::init(needed_spacing);
  
  
  bool update_A_spacing = false; 
  bool update_B_spacing = false;
  #ifdef NTTMUL
	if(a.CRTSPACING != needed_spacing)
	  a.update_crt_spacing(needed_spacing);
	if(b.CRTSPACING != needed_spacing)
	  b.update_crt_spacing(needed_spacing);
  #elif defined(CUFFTMUL)
  if(a.get_crt_spacing() != needed_spacing){
  	// if(!a.get_device_updated())
  		// a.update_crt_spacing(needed_spacing);
  	// else
	  	update_A_spacing = true;
  	
  }
  if(b.get_crt_spacing() != needed_spacing){
  	// if(!b.get_device_updated())
  		// b.update_crt_spacing(needed_spacing);
  	// else
	  	update_B_spacing = true;
  }
  #endif

  #ifdef VERBOSE
  std::cout << "Mul with CRTSPACING " << needed_spacing << std::endl;
  // std::cout << "this: " << a.to_string() << std::endl;
  // std::cout << "other " << b.to_string() << std::endl;
  #endif

  // Apply CRT and copy data to global memory, if needed
  // #pragma omp sections
  {
      // #pragma omp section
      {	
		#ifdef VERBOSE
		std::cout << "a" << std::endl;
		#endif
		if(!a.get_device_updated()){
			a.update_device_data(2);
		}

      }
      // #pragma omp section
      {
		#ifdef VERBOSE
		std::cout << "b" << std::endl;
		#endif
		if(!b.get_device_updated()){
			b.update_device_data(2);
		}
      }
  }
  start = get_cycles();

  cuyasheint_t *d_result;
  if(a.get_crt_spacing() > 0 && b.get_crt_spacing() > 0)
	  d_result = CUDAFunctions::callPolynomialMul(a.get_stream(),
															a.get_device_crt_residues(),
															update_A_spacing,
															a.get_crt_spacing(),
															b.get_device_crt_residues(),
															update_B_spacing,
															b.get_crt_spacing(),
															needed_spacing,
															a.CRTPrimes.size());
  else
  	d_result = NULL;

  end = get_cycles();
  // std::cout << (end-start) << " cycles to multiply" << std::endl;
  P c = P(a.get_mod(),a.get_phi(),needed_spacing);
  if(d_result != NULL){
	  c.set_device_crt_residues(d_result);
	  c.set_host_updated(false);
	  c.set_device_updated(true);

	  // std::cout << " resultado da multiplicação: " << c.to_string() << std::endl;
	  c.reduce();
	  c %= a.get_mod();
  }
  // cudaDeviceSynchronize();
  // std::cout << (end-start) << " cycles for polynomial x polynomial mul" << std::endl;
  return c;
}
#endif
