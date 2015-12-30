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

bn_t CUDAFunctions::M;
bn_t* CUDAFunctions::Mpis;
cuyasheint_t* CUDAFunctions::invMpis;

////////////////////////
// Auxiliar functions //
////////////////////////

__host__ __device__ int max_d(int a,int b){
	return (a >= b)*a + (b > a)*b;
}

__host__ __device__ int min_d(int a,int b){
	return (a <= b)*a + (b < a)*b;
}

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

__device__ void bn_new_d(bn_t *a){
  a->used = 0;
  a->alloc = STD_BNT_ALLOC;
  a->sign = BN_POS;
  cudaMalloc(&a->dp,a->alloc*sizeof(cuyasheint_t));
}

__host__ void bn_free(bn_t *a){
  a->used = 0;
  a->alloc = 0;
  
  cudaError_t result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);
  result = cudaFree(a->dp);
  assert(result == cudaSuccess);

}

__host__ __device__ int bn_cmpn_low(const cuyasheint_t *a, const cuyasheint_t *b, int size) {
	int i, r;

	a += (size - 1);
	b += (size - 1);

	r = CMP_EQ;
	for (i = 0; i < size; i++, --a, --b) {
		if (*a != *b && r == CMP_EQ) {
			r = (*a > *b ? CMP_GT : CMP_LT);
		}
	}
	return r;
}


__host__ __device__ int bn_cmp_abs(const bn_t *a, const bn_t *b) {
	if (a->used > b->used) {
		return CMP_GT;
	}

	if (a->used < b->used) {
		return CMP_LT;
	}

	return bn_cmpn_low(a->dp, b->dp, a->used);
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

__device__ void bn_grow_d(bn_t *a,const unsigned int new_size){
  // We expect that a->alloc <= new_size
  if((unsigned int)a->alloc >= new_size)
  	return;

  cudaMalloc(&a->dp+a->alloc,new_size*sizeof(cuyasheint_t));
  a->alloc = new_size;

}

////////////////
// Operators //
//////////////

// Mod
__host__ __device__ cuyasheint_t bn_mod1_low(const cuyasheint_t *a,
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
__host__ __device__ cuyasheint_t bn_mul1_low(cuyasheint_t *c,
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
									cuyasheint_t m
									){
	/**
	 * http://stackoverflow.com/a/18680280/1541615
	 */
    uint64_t res = 0;
    uint64_t temp_b;

    /* Only needed if b may be >= m */
    if (b >= m) {
        if (m > UINT64_MAX / 2u)
            b -= m;
        else
            b %= m;
    }

    while (a != 0) {
        if (a & 1) {
            /* Add b to res, modulo m, without overflow */
            if (b >= m - res) /* Equiv to if (res + b >= m), without overflow */
                res -= m;
            res += b;
        }
        a >>= 1;

        /* Double b, modulo m */
        temp_b = b;
        if (b >= m - b)       /* Equiv to if (2 * b >= m), without overflow */
            temp_b -= m;
        b += temp_b;
    }
	*result = res;
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
__host__ __device__ cuyasheint_t bn_addn_low(cuyasheint_t *c,
									cuyasheint_t *a,
									cuyasheint_t *b,
									const int size
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

/**
 * [bn_add1_low description]
 * @param  c     [description]
 * @param  a     [description]
 * @param  digit [description]
 * @param  size  [description]
 * @return       [description]
 */
__host__ __device__ cuyasheint_t bn_add1_low(cuyasheint_t *c, const cuyasheint_t *a, cuyasheint_t digit, int size) {
	int i;
	register cuyasheint_t carry, r0;

	carry = digit;
	for (i = 0; i < size && carry; i++, a++, c++) {
		r0 = (*a) + carry;
		carry = (r0 < carry);
		(*c) = r0;
	}
	for (; i < size; i++, a++, c++) {
		(*c) = (*a);
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
						const unsigned int N,
						const unsigned int NPolis
						){
	/**
	 * This function should be executed with N*Npolis threads. 
	 * Each thread computes one coefficient of each residue of d_polyCRT
	 *
	 * x should be an array of N elements
	 * d_polyCRT should be an array of N*NPolis elements
	 */
	
	/**
	 * tid: thread id
	 * cid: coefficient id
	 * rid: residue id
	 */
	const unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;
	const unsigned int cid = tid & (N-1);
	const unsigned int rid = tid / N;

	if(tid < N*NPolis)
		// Computes x mod pi
		d_polyCRT[tid] = bn_mod1_low(	x[cid].dp,
										x[cid].used,
										CRTPrimesConstant[rid]
										);
	
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
						const unsigned int N,
						const unsigned int NPolis,
						const bn_t M,
						const bn_t *Mpis,
						const cuyasheint_t *invMpis,
						bn_t *inner_results
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

	 	bn_zero(&poly[cid]);
	 	for(unsigned int rid = 0; rid < NPolis;rid++){
				volatile cuyasheint_t carry;
	 			cuyasheint_t x;

	 			// Get a prime
	 			cuyasheint_t pi = (cuyasheint_t)CRTPrimesConstant[rid];
	 	
	 			// Computes the inner result
	 			bn_t inner_result = inner_results[cid];
	 			bn_zero(&inner_result);
	 	
	 			/**
	 			 * Inner
	 			 */
	 			bn_64bits_mulmod(	&x,
	 								invMpis[rid],
	 								d_polyCRT[cid + rid*N],
	 								pi);

	 			// Adjust available words in inner_result
 				if(inner_result.alloc < Mpis[rid].used+1)
 					bn_grow_d(&inner_result,Mpis[rid].alloc+1);

	 			carry = bn_mul1_low(inner_result.dp,
		 					     	Mpis[rid].dp,
		 					     	x,
		 					     	Mpis[rid].used);
 				
 				inner_result.used = Mpis[rid].used;
	 			if(carry){
 					inner_result.dp[inner_result.used] = carry;	
	 				inner_result.used++;	 				
 				}

 				/**
 				 * Accumulate
 				 */

				bn_t a = ( bn_cmp_abs(&poly[cid],&inner_result) == CMP_GT? poly[cid] : inner_result );
				bn_t b = ( bn_cmp_abs(&poly[cid],&inner_result) == CMP_LT? poly[cid] : inner_result);

				int max = a.used;
				int min = b.used;

				/* Grow the result. */
				bn_grow_d(&poly[cid], max);

				if (a.used == b.used) {
					carry = bn_addn_low(poly[cid].dp, a.dp, b.dp, max);
				} else {
					carry = bn_addn_low(poly[cid].dp, a.dp, b.dp, min);
					carry = bn_add1_low(poly[cid].dp + min, a.dp + min, carry, max - min);
				}

				poly[cid].used = max;
				if (carry) {
					bn_grow_d(&poly[cid], max + 1);
					poly[cid].dp[max] = carry;
					poly[cid].used++;
				}

 				__syncthreads();
	 		}

	 ////////////////////////////////////////////////
	 // To-do: Modular reduction of poly[cid] by M //
	 ////////////////////////////////////////////////
	 }

}

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
	
	cudaError_t result = cudaDeviceSynchronize();
	assert(result == cudaSuccess);

	cuCRT<<<gridDim,blockDim,0,stream>>>(d_polyCRT,coefs,N,NPolis);
}
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

	bn_t *inner_results;
	cudaMallocManaged((void**)&inner_results, N*sizeof(bn_t));

	cudaError_t result = cudaDeviceSynchronize();
	assert(result == cudaSuccess);

	for(unsigned int i =0; i < N; i++)
		bn_new(&inner_results[i]);

	result = cudaDeviceSynchronize();
	assert(result == cudaSuccess);
  	
	cuICRT<<<gridDim,blockDim,0,stream>>>(	coefs,
											d_polyCRT,
											N,
											NPolis,
											CUDAFunctions::M,
											CUDAFunctions::Mpis,
											CUDAFunctions::invMpis,
											inner_results);
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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /////////////////
    // Copy primes //
    /////////////////
    cudaError_t result = cudaMemcpyToSymbolAsync ( CRTPrimesConstant,
                                              &(Polynomial::CRTPrimes[0]),
                                              Polynomial::CRTPrimes.size()*sizeof(cuyasheint_t),
                                              0,
	                                           cudaMemcpyHostToDevice,
                                              stream
                                            );
    assert(result == cudaSuccess);

    ////////////
    // Copy M //
    ////////////

    // if(M)
    	// cudaFree(M);
    // cudaMallocManaged((void**)&M,sizeof(bn_t));
    // get_words(M,Polynomial::CRTProduct);

    //////////////
    // Copy Mpi //
    //////////////
    
    if(CUDAFunctions::Mpis)
    	cudaFree(CUDAFunctions::Mpis);
    cudaMallocManaged((void**)&CUDAFunctions::Mpis,Polynomial::CRTPrimes.size()*sizeof(cuyasheint_t));
    for(unsigned int i = 0; i < Polynomial::CRTPrimes.size();i++)
    	get_words(&CUDAFunctions::Mpis[i],Polynomial::CRTMpi[i]);

    /////////////////
    // Copy InvMpi //
    /////////////////

    if(CUDAFunctions::invMpis)
    	cudaFree(CUDAFunctions::invMpis);
    result = cudaMalloc((void**)&CUDAFunctions::invMpis,Polynomial::CRTPrimes.size()*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

	result = cudaMemcpyAsync(CUDAFunctions::invMpis,
							&Polynomial::CRTInvMpi[0],
							Polynomial::CRTPrimes.size()*sizeof(cuyasheint_t),
							cudaMemcpyHostToDevice
							);
    assert(result == cudaSuccess);


  }else{
    throw "Too many primes.";
  }
}