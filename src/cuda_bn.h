#ifndef CUDA_BN_H
#define CUDA_BN_H

#include <NTL/ZZ.h>
#include "settings.h"

NTL_CLIENT 

//////////////////////
// Big number's stuff //
//////////////////////

#define BN_POS      (0)
#define BN_NEG      (1)
#define CMP_LT 		-1
#define CMP_GT 		 1
#define CMP_EQ		 0

#define BN_DIGIT WORD
typedef struct bn_st{
	int alloc = 0;
	int used = 0;
	int sign = BN_POS;
	cuyasheint_t *dp = NULL;
} bn_t;

__host__  void bn_new(bn_t *a);
__host__ __device__ void bn_zero(bn_t *a);
__host__ __device__ void bn_set_dig(bn_t *a, cuyasheint_t digit);
__host__  void bn_free(bn_t *a);
__host__ void bn_grow(bn_t *a,const unsigned int new_size);
__host__ __device__ cuyasheint_t bn_mod1_low(const cuyasheint_t *a,
									const int size,
									const uint32_t b);
__host__ __device__ cuyasheint_t bn_addn_low(cuyasheint_t *c,
									cuyasheint_t *a,
									cuyasheint_t *b,
									const int size
									);
__host__ __device__ cuyasheint_t bn_subn_low(	cuyasheint_t * c,
												const cuyasheint_t * a,
												const cuyasheint_t * b, 
												int size);
__host__ __device__ cuyasheint_t bn_sub1_low(	cuyasheint_t *c,
												const cuyasheint_t *a,
												cuyasheint_t digit,
												int size);
__host__ __device__ cuyasheint_t bn_mul1_low(cuyasheint_t *c,
									const cuyasheint_t *a,
									cuyasheint_t digit,
									int size);
__host__ __device__ int bn_cmpn_low(const cuyasheint_t *a,
									const cuyasheint_t *b,
									int size);
__host__ __device__ int bn_cmp_abs(	const bn_t *a,
									const bn_t *b);
__host__ void callCuModN(bn_t * c, const bn_t * a,int NCoefs,
		const cuyasheint_t * m, int sm, const cuyasheint_t * u, int su,
		cudaStream_t stream);
__host__ ZZ get_ZZ(bn_t *a);
__host__ __device__ int max_d(int a,int b);
__host__ __device__ int min_d(int a,int b);
__host__ void callTestData(bn_t *coefs,int N);
void callCRT(bn_t *coefs,const int used_coefs,cuyasheint_t *d_polyCRT,const int N, const int NPolis,cudaStream_t stream);
void callICRT(bn_t *coefs,cuyasheint_t *d_polyCRT,const int N, const int NPolis,cudaStream_t stream);


#endif
