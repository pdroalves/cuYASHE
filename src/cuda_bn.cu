#include "cuda_bn.h"
#include "settings.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "cuda_functions.h"


cuyasheint_t bn_mod1_low(const cuyasheint_t *a, const int size, const cuyasheint_t b) {
	// Computes a % b
	
	dcuyasheint_t w;
	cuyasheint_t r;
	int i;

	w = 0;
	for (i = size - 1; i >= 0; i--) {
		w = (w << ((dcuyasheint_t)BN_DIGIT)) | ((dcuyasheint_t)a[i]);

		if (w >= b) {
			r = (cuyasheint_t)(w / b);
			w -= ((dcuyasheint_t)r) * ((dcuyasheint_t)b);
		} else {
			r = 0;
		}
	}
	return (cuyasheint_t)w;
}

/**
 * @d_polyCRT - output: array of residual polynomials
 * @p - input: array of coefficients
 * @ N - input: qty of coefficients
 * @NPolis - input: qty of primes/residual polynomials
 */
__global__ void CRT(cuyasheint_t *d_polyCRT, const bn_t *p,const int unsigned N,const unsigned int NPolis){
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	const int cid = tid & (constant -1 ); // We expect that N is power of two

	if(tid < N*NPolis){

		// pid <=> prime's id
		for(unsigned int pid = 0; pid < NPolis; pid++)
			// Load this thread's coefficient
			// Computes x mod pi
			d_polyCRT[cid + pid*N] = bn_mod1_low(&result,p[cid],CRTPrimesConstant[pid]);

	}
}