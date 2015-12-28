#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "cuda_bn.h"
#include "settings.h"
#include "cuda_functions.h"
#include "polynomial.h"

__constant__ cuyasheint_t CRTPrimesConstant[MAX_PRIMES_ON_C_MEMORY];

//////////////////////////
// CRT global variables //
//////////////////////////

__device__ bn_t* M;
__device__ bn_t* Mpis;
__device__ cuyasheint_t *invMpis;

////////////////////////
// Auxiliar functions //
////////////////////////


__host__ __device__ void dv_zero(cuyasheint_t *a, int digits) {
	int i;
 
	// if (digits > DV_DIGS) {
	// 	std::cout << "ERR_NO_VALID" << std::endl;
	// 	exit(1);
	// }
	cudaError_t result = cudaDeviceSynchronize();
	assert(result == cudaSuccess);

	
	for (i = 0; i < digits; i++, a++)
		(*a) = 0;

	return;
}

/**
 * Set a big number struct to zero
 * @param a operand
 */
__host__ __device__ void bn_zero(bn_t *a) {
	a->sign = BN_POS;
	a->used = 1;
	dv_zero(a->dp, a->alloc);
}

/**
 * Set a big number to digit
 * @param a     input: big number
 * @param digit input: digit
 */
__host__ __device__ void bn_set_dig(bn_t *a, cuyasheint_t digit) {
	cudaError_t result = cudaDeviceSynchronize();
	assert(result == cudaSuccess);

	bn_zero(a);	
	a->dp[0] = digit;
	a->used = 1;
	a->sign = BN_POS;
}

__host__ void bn_new(bn_t *a){
  a->used = 0;
  a->alloc = STD_BNT_ALLOC;
  a->sign = BN_POS;
  cudaMallocManaged(&a->dp,a->alloc*sizeof(cuyasheint_t));
}

__host__ void bn_free(bn_t *a){
  a->used = 0;
  a->alloc = 0;
  
  cudaError_t result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);
  result = cudaFree(a->dp);
  assert(result == cudaSuccess);

}

/**
 * Increase the allocated memory for a bn_t object.
 * @param a        input/output:operand
 * @param new_size input: new_size for dp
 */
__host__ void bn_grow(bn_t *a,const unsigned int new_size){
  // We expect that a->alloc <= new_size
  if((unsigned int)a->alloc <= new_size)
  	return;

  cudaMallocManaged(&a->dp+a->alloc,new_size*sizeof(cuyasheint_t));
  a->alloc = new_size;

}


////////////////
// Operators //
//////////////

// Mod
__device__ cuyasheint_t bn_mod1_low(const cuyasheint_t *a,
									const int size,
									const cuyasheint_t b) {
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

// Multiply 

/**
 * Computes a*digit
 * @param  c     output: result
 * @param  a     input: many-words first operand
 * @param  digit input: one-word second operand
 * @param  size  input: number of words in a
 * @return       output: result's last word
 */
__device__ cuyasheint_t bn_mul1_low(cuyasheint_t *c,
									const cuyasheint_t *a,
									cuyasheint_t digit,
									int size) {
	int i;
	cuyasheint_t carry;
	dcuyasheint_t r;

	carry = 0;
	for (i = 0; i < size; i++, a++, c++) {
		/* Multiply the digit *tmpa by b and accumulate with the previous
		 * result in the same columns and the propagated carry. */
		r = (dcuyasheint_t)(carry) + (dcuyasheint_t)(*a) * (dcuyasheint_t)(digit);
		/* Increment the column and assign the result. */
		*c = (cuyasheint_t)r;
		/* Update the carry. */
		carry = (cuyasheint_t)(r >> (dcuyasheint_t)BN_DIGIT);
	}
	
	// Adds carry as last word
	c++;
	*c = (cuyasheint_t)carry;

	return carry;
}

/**
 * Computes 64bits a*b mod c
 * @param result       output: result
 * @param a            input: first 64 bits operand
 * @param b            input: second 64 bits operand 
 * @param c 		   input: module
 */
__device__ void bn_64bits_mulmod(cuyasheint_t *result,
									cuyasheint_t a,
									cuyasheint_t b,
									cuyasheint_t c
									){

	uint64_t w;
	uint64_t r;
	const int size = 2;
	
	/**
	 * Mul
	 */
	const uint64_t A[2] = {a*b,__umul64hi(a,b)};

	/**
	 * Modular reduction
	 */
	w = 0;
	for (int i = size - 1; i >= 0; i--){
		w = (w << ((uint64_t)32)) | ((uint64_t)A[i]);

		if (w >= b) {
			r = (uint32_t)(w / b);
			w -= ((uint64_t)r) * ((uint64_t)b);
		} else {
			r = 0;
		}
	}

	*result = r;
}

// Add

/**
 * Computes a+b
 * @param  c    output: result
 * @param  a    input: many-words first operand
 * @param  b    input: many-words second operand
 * @param  size input: number of words to add
 * @return      output: result's last word
 */
__device__ cuyasheint_t bn_addn_low(cuyasheint_t *c,
									const cuyasheint_t *a,
									const cuyasheint_t *b,
									int size
									) {
	int i;
	register cuyasheint_t carry, c0, c1, r0, r1;

	carry = 0;
	for (i = 0; i < size; i++, a++, b++, c++) {
		r0 = (*a) + (*b);
		c0 = (r0 < (*a));
		r1 = r0 + carry;
		c1 = (r1 < r0);
		carry = c0 | c1;
		(*c) = r1;
	}
	return carry;
}

/////////
// CRT //
/////////

/**
 * @d_polyCRT - output: array of residual polynomials
 * @x - input: array of coefficients
 * @ N - input: qty of coefficients
 * @NPolis - input: qty of primes/residual polynomials
 */
__global__ void cuCRT(	cuyasheint_t *d_polyCRT,
						const bn_t *x,
						const int unsigned N,
						const unsigned int NPolis
						){
	/**
	 * This function should be executed with N*Npolis threads. 
	 * Each thread computes one coefficient of each residue of d_polyCRT
	 */
	
	/**
	 * tid: thread id
	 * cid: coefficient id
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	const int cid = tid & (N -1 ); // We expect that N is a power of two

	if(tid < N*NPolis){

		// pid == tid <=> prime's id
		// Load this thread's coefficient
		// Computes x mod pi
		d_polyCRT[cid + tid*N] = bn_mod1_low(	x[cid].dp,
												x[cid].used,
												CRTPrimesConstant[tid]
												);
	}
}	

/**
 * cuICRT computes ICRT on GPU
 * @param poly      output: An array of coefficients 
 * @param d_polyCRT input: The CRT residues
 * @param N         input: Number of coefficients
 * @param NPolis    input: Number of residues
 */
__global__ void cuICRT(	bn_t *poly,
						const cuyasheint_t *d_polyCRT,
						const int unsigned N,
						const unsigned int NPolis,
						cuyasheint_t *CRTPrimes
						){
	/**
	 * This function should be executed with N threads.
	 * Each thread j computes a Mpi*( invMpi*(value) % pi) and adds to poly[j]
	 */
	
	/**
	 * tid: thread id
	 * cid: coefficient id
	 * rid: residue id
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;
	const int cid = tid & (N -1 ); // We expect that N is a power of two

	 if(tid < N){

	 	for(unsigned int rid = 0; rid < NPolis;rid++){
	 			// Get a prime
	 			cuyasheint_t pi = CRTPrimes[rid];
	 	
	 			// Computes the inner result
	 			bn_t inner_result;
	 			bn_zero(&inner_result);
	 			cuyasheint_t x;
	 	
	 			bn_64bits_mulmod(	&x,
	 								invMpis[rid],
	 								d_polyCRT[tid],
	 								pi);
	 			bn_mul1_low(	inner_result.dp,
	 					     	Mpis[rid].dp,
	 					     	x,
	 					     	Mpis[rid].used);
	 			
	 			bn_addn_low(poly[cid].dp,
							poly[cid].dp,
							inner_result.dp,
							inner_result.used
							);
	 		}

	 ////////////////////////////////////////////////
	 // To-do: Modular reduction of poly[cid] by M //
	 ////////////////////////////////////////////////
	 }

}


// __global__ void cuCRT(	cuyasheint_t *d_polyCRT,
// 						const bn_t *x,
// 						const int unsigned N,
// 						const unsigned int NPolis
// 						){
	/**
	 * This function should be executed with N*Npolis threads. 
	 * Each thread computes one coefficient of each residue of d_polyCRT
	 */
	
void crt(bn_t *coefs,cuyasheint_t *d_polyCRT,const int N, const int NPolis,cudaStream_t stream){

	const int size = N*NPolis;
	int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? 
			size/ADDBLOCKXDIM : 
			size/ADDBLOCKXDIM + 1);
	dim3 gridDim(ADDGRIDXDIM);
	dim3 blockDim(ADDBLOCKXDIM);
	
	cuCRT<<<gridDim,blockDim,1,stream>>>(d_polyCRT,coefs,N,NPolis);
}

// __global__ void cuICRT(	bn_t *poly,
// 						const cuyasheint_t *d_polyCRT,
// 						const int unsigned N,
// 						const unsigned int NPolis
// 						){
	/**
	 * This function should be executed with N threads.
	 * Each thread j computes a Mpi*( invMpi*(value) % pi) and adds to poly[j]
	 */

void icrt(bn_t *coefs,cuyasheint_t *d_polyCRT,const int N, const int NPolis,cudaStream_t stream){

	const int size = N;
	int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? 
			size/ADDBLOCKXDIM : 
			size/ADDBLOCKXDIM + 1);
	dim3 gridDim(ADDGRIDXDIM);
	dim3 blockDim(ADDBLOCKXDIM);
	///////////
	// To-do //
	///////////
}

__host__ void  CUDAFunctions::write_crt_primes(){

  #ifdef VERBOSE
  std::cout << "primes: "<< std::endl;
  for(unsigned int i = 0; i < Polynomial::CRTPrimes.size();i++)
    std::cout << Polynomial::CRTPrimes[i] << " ";
  std::cout << std::endl;
  #endif
  
  // Choose what memory will be used to story CRT Primes
  if(Polynomial::CRTPrimes.size() < MAX_PRIMES_ON_C_MEMORY){
    
    #ifdef VERBOSE
    std::cout << "Writting CRT Primes to GPU's constant memory" << std::endl;
    #endif

    cudaError_t result = cudaMemcpyToSymbol ( CRTPrimesConstant,
                                              &(Polynomial::CRTPrimes[0]),
                                              Polynomial::CRTPrimes.size()*sizeof(cuyasheint_t)
                                            );
    assert(result == cudaSuccess);
  }else{
    throw "Too many primes.";
  }
}