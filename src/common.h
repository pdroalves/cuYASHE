#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include "settings.h"
#include <NTL/ZZ.h>
#include <NTL/ZZ_pEX.h>
#include <NTL/ZZ_pXFactoring.h>

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
			if(a->CRTSPACING != b->CRTSPACING){
			  int new_spacing = std::max(a->CRTSPACING,b->CRTSPACING);
			  a->update_crt_spacing(new_spacing);
			  b->update_crt_spacing(new_spacing);
			}

			#ifdef VERBOSE
			std::cout << "Adding:" << std::endl;
			// std::cout << "this: " << a->to_string() << std::endl;
			// std::cout << "other " << b->to_string() << std::endl;
			#endif

			// Apply CRT and copy data to global memory, if needed
			// #pragma omp parallel sections num_threads(2)
			{
			    // #pragma omp section
			    {

			        if(!a->get_device_updated()){
			          a->crt();
			          a->update_device_data();
			        }

			    }
			    // #pragma omp section
			    {
			        if(!b->get_device_updated()){
			            b->crt();
			            b->update_device_data();
			        }
			    }
			}


			cuyasheint_t *d_result = CUDAFunctions::callPolynomialAddSub(a->get_stream(),a->get_device_crt_residues(),b->get_device_crt_residues(),(int)(a->CRTSPACING*P::CRTPrimes.size()),ADD);

			P c(a->get_mod(),a->get_phi(),a->CRTSPACING);
			c.set_device_crt_residues(d_result);
			c.set_host_updated(false);
			c.set_device_updated(true);
			cudaDeviceSynchronize();
			return c;
	#ifdef MAYADDONCPU
		}
	#endif
}

template <class P>
void common_addition_inplace(P *a,P *b){
	// P should be Polynomial or Ciphertext
	// Store result in polynomial a
	#ifdef MAYADDONCPU
		if(!a->get_device_updated() && !b->get_device_updated()){
		// CPU add
		#ifdef VERBOSE
		std::cout << "Operator+= on CPU" << std::endl;
		#endif
		a->CPUAddition(&b);
		return;
		}else{
	#endif

	#ifdef VERBOSE
	std::cout << "Operator+= on GPU" << std::endl;
	#endif

	// Check align
	if(a->CRTSPACING != b->CRTSPACING){
		int new_spacing = std::max(a->CRTSPACING,b->CRTSPACING);
		a->update_crt_spacing(new_spacing);
		b->update_crt_spacing(new_spacing);
	}

	#ifdef VERBOSE
	std::cout << "Adding:" << std::endl;
	// std::cout << "this: " << a->to_string() << std::endl;
	// std::cout << "other " << b->to_string() << std::endl;
	#endif

	// Apply CRT and copy data to global memory, if needed
	#pragma omp parallel sections num_threads(2)
	{
		#pragma omp section
		{

			if(!a->get_device_updated()){
				a->crt();
				a->update_device_data();
			}

		}
		#pragma omp section
		{
			if(!b->get_device_updated()){
				b->crt();
				b->update_device_data();
			}
		}
	}


	CUDAFunctions::callPolynomialAddSubInPlace(a->get_stream(),a->get_device_crt_residues(),b->get_device_crt_residues(),(int)(a->CRTSPACING*P::CRTPrimes.size()),ADD);

	a->set_host_updated(false);
	a->set_device_updated(true);
	cudaDeviceSynchronize();
	#ifdef MAYADDONCPU
	}
	#endif
}

template <class P>
P common_multiplication(P *a, P *b){

  // Check align
  int new_spacing = pow(2,ceil(log2(a->deg()+b->deg())));
  
  if(new_spacing < CUDAFunctions::N)
	new_spacing = CUDAFunctions::N;
	else if(new_spacing != CUDAFunctions::N){
	// Re-compute W matrix
	CUDAFunctions::init(new_spacing);
  }

  a->update_crt_spacing(new_spacing);
  b->update_crt_spacing(new_spacing);

  #ifdef VERBOSE
  std::cout << "Mul:" << std::endl;
  // std::cout << "this: " << a->to_string() << std::endl;
  // std::cout << "other " << b->to_string() << std::endl;
  #endif

  // Apply CRT and copy data to global memory, if needed
  #pragma omp sections
  {
      #pragma omp section
      {

          if(!a->get_device_updated()){
            a->crt();
            a->update_device_data(2);
          }

      }
      #pragma omp section
      {
          if(!b->get_device_updated()){
              b->crt();
              b->update_device_data(2);
          }
      }
  }

  cuyasheint_t *d_result = CUDAFunctions::callPolynomialMul(a->get_stream(),a->get_device_crt_residues(),b->get_device_crt_residues(),a->CRTSPACING,a->CRTPrimes.size());

  P c(a->get_mod(),a->get_phi(),a->CRTSPACING);
  c.set_device_crt_residues(d_result);
  c.set_host_updated(false);
  c.set_device_updated(true);
  // cudaDeviceSynchronize();
  return c;
}
#endif
