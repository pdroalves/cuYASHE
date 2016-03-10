#include "cuda_ciphertext.h"

template <int WORDLENGTH>
/**
 * cuWordecomp computes de word decomposition of every coefficient in 32 bit words.
 * The coefficients are destroyed inside the kernel
 * @param P   [description]
 * @param a   [description]
 * @param lwq [description]
 * @param N   [description]
 */
__global__ void cuWordecomp(bn_t **P,bn_t *a,int lwq, int N){
  printf("Nothing to do");
}
/**
 * Computes WordDecomp for W = 2^32
 * @param P   [description]
 * @param a   [description]
 * @param lwq [description]
 */
template<>
__global__ void cuWordecomp<32>(bn_t **P,bn_t *a,int lwq, int N){
	/**
	 * This kernel should be executed by lwq thread
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < N){
		// P_i = [a_i]_w
		bn_t c = a[tid];
		// c will be modified
		int j = 0;
		int c_index = 0;
     	// c.used = get_used_index(c.dp,c.used) + 1;
		while(c_index < c.used){
			P[j][tid].dp[0] = (uint32_t)(c.dp[c_index]);
			if(c.dp[c_index]>>32)
				P[j+1][tid].dp[0] = (uint32_t)(c.dp[c_index]>>32);
			c_index++;
			j += 2;
		}
	}
}

/**
 * Computes WordDecomp for W = 2^64
 * @param P   [description]
 * @param a   [description]
 * @param lwq [description]
 */
template<>
__global__ void cuWordecomp<64>(bn_t **P,bn_t *a,int lwq, int N){	/**
	 * This kernel should be executed by lwq thread
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < N){
		// P_i = [a_i]_w
		bn_t c = a[tid];
		// c will be modified
		cuyasheint_t dp[STD_BNT_WORDS_ALLOC];
		c.dp = dp;
		bn_copy(&c,&a[tid]);

		int j = 0;
		while(c.used > 0){
			P[j][tid].dp[0] = c.dp[0] & 18446744073709551615L;
			bn_rshb_low(c.dp, c.dp, c.used, 64);
			c.used -= 1;
			j++;
		}
	}
}


void callCuWordecomp(	dim3 gridDim, 
						dim3 blockDim, 
						cudaStream_t stream, 
						int WORDLENGTH, 
						bn_t **d_P, 
						bn_t *a, 
						int lwq, 
						int N){
	if(WORDLENGTH == 32)
		cuWordecomp<32><<<gridDim,blockDim,0,stream>>>(d_P,a,lwq, N);
	else if(WORDLENGTH == 64)
		cuWordecomp<64><<<gridDim,blockDim,0,stream>>>(d_P,a,lwq, N);
	else
		throw "Unknown WORDLENGTH";
	cudaError_t result = cudaGetLastError();
	assert(result == cudaSuccess);
}

/**
 * Convert an array with 64 bits elements to a array of 32 bits elements
 * @param a [description]
 * @param b [description]
 * @param n [description]
 */
__host__ __device__ void convert_64_to_32(uint32_t *a,uint64_t *b,int n){
	// This function supposes that b has n positions and a has 2*n positions
	for(int i = 0; i < n; i += 1){
		a[2*i] = (uint32_t)b[i]; // First word
		a[2*i+1] = (uint32_t)(b[i] >> 32); //Second word
	}
}

__host__ __device__ void convert_32_to_64(uint64_t *a, uint32_t *b, int n){
	// This function supposes that b has 2*n positions and a has n positions
	for(int i = 0; i < 2*n; i += 2){
		a[i/2] = b[i] | b[i+1];
	}
}

/**
 * Computes g/q and g%q and set "output" according the result   
 *
 * bn_divn_low: Divides a digit vector by another digit vector. Computes c = floor(a / b) and d = a mod b. 
 * The dividend and the divisor are destroyed inside the function. So, this function expect that g won't be
 * used again.
 * 
 * @param output [description]
 * @param g      [description]
 * @param q      [description]
 * @param N
 */
__global__ void cuCiphertextMulAux(	bn_t *P, 
									bn_t *g, 
									uint32_t *q,
									int q_used, 
									bn_t qDiv2,
									int N){
	/**
	 * This kernel should be executed with N threads
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < N){
		bn_t coef = g[tid];
		uint32_t q_digits[2*STD_BNT_WORDS_ALLOC];
		uint32_t coef_digits[2*STD_BNT_WORDS_ALLOC];
		uint64_t diff_dp[STD_BNT_WORDS_ALLOC];
		uint32_t diff_digits[2*STD_BNT_WORDS_ALLOC];
		bn_t output = P[tid];
		uint32_t output_digits[2*STD_BNT_WORDS_ALLOC];

		#pragma unroll
		for(int i = 0; i < 2*STD_BNT_WORDS_ALLOC; i++){
			output_digits[i] = 0;
			diff_digits[i] = 0;
			q_digits[i] = 0;
		}

		// q will be destroyed in bn_divn either
		for(int i = 0; i < q_used; i++)
			q_digits[i] = q[i];

		int n = coef.used;

		convert_64_to_32(coef_digits, coef.dp, n);

		bn_divn_low(output_digits, diff_digits, coef_digits, 2*n, q_digits, q_used);

		convert_32_to_64(output.dp, output_digits, n);
		convert_32_to_64(diff_dp, diff_digits, n);

		bn_t diff;
		diff.alloc = STD_BNT_WORDS_ALLOC;
		diff.sign = BN_POS;
		diff.dp = diff_dp;
		// Check if diff > q/2
		int cmp = bn_cmp_abs(&diff, &qDiv2);
		if(cmp == CMP_GT)
			// I suppose that output.dp[0] +1 < 2**64 
			bn_add1_low(output.dp, output.dp, 1, 1);
	}
}

__global__ void cuCiphertextMulAuxMersenne( bn_t *g, 
											int q_bits,
											int N){
	/**
	 * This kernel should be executed with N threads
	 *
	 * It supposes that q is a mersenne prime
	 */
	const int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if(tid < N){
		// Input
		bn_t coef = g[tid];

		// q_bits right shift
		uint64_t carry = bn_rshb_low(coef.dp,
									coef.dp,
									coef.used,
									q_bits);
		if(carry)
			bn_rshb_low(coef.dp,
						coef.dp,
						coef.used,
						q_bits);
		coef.dp -= (q_bits / WORD) + (q_bits % WORD != 0);

	}
}

/**
 * Computes the second part of a ciphertext multiplication
 * @param P      [description]
 * @param g      [description]
 * @param q      [description]
 * @param N      [description]
 * @param stream [description]
 */
__host__ void callCiphertextMulAux(bn_t *P, bn_t *g, ZZ q,int N, cudaStream_t stream){
	const int size = N;
	const int ADDGRIDXDIM = (size%128 == 0? size/128 : size/128 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(128);

	// Computes bn_t q and bn_t q/2
	bn_t Q;
	get_words_host(&Q,q);
	bn_t QDiv2;
	get_words_host(&QDiv2,q/2);

	// Computes Q with 32 bits words
	uint32_t *h_Q = (uint32_t*) malloc (2*Q.used*sizeof(uint32_t));
	uint32_t *d_Q;
	cudaError_t result = cudaMalloc((void**)&d_Q,2*Q.used*sizeof(uint32_t));
	assert(result == cudaSuccess);

	convert_64_to_32(h_Q, Q.dp, Q.used);

	result = cudaMemcpyAsync(d_Q, h_Q, 2*Q.used*sizeof(uint32_t),cudaMemcpyHostToDevice,stream);
	assert(result == cudaSuccess);
	// Temporary polynomial
	Polynomial diff(N);

	cuCiphertextMulAux<<<gridDim, blockDim, 0, stream>>>(P, g, d_Q, 2*Q.used, QDiv2, N);
	result = cudaGetLastError();
	assert(result == cudaSuccess);

	free(h_Q);
	result = cudaFree(d_Q);
	assert(result == cudaSuccess);
}

__host__ void callCiphertextMulAuxMersenne(bn_t *g, ZZ q,int N, cudaStream_t stream){

	const int size = N;
	const int ADDGRIDXDIM = (size%128 == 0? size/128 : size/128 + 1);
	const dim3 gridDim(ADDGRIDXDIM);
	const dim3 blockDim(128);

	cuCiphertextMulAuxMersenne<<<gridDim, blockDim,0, stream>>>(g,NTL::NumBits(q),N);
	cudaError_t result = cudaGetLastError();
	assert(result == cudaSuccess);

}