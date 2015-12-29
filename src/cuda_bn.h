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

typedef uint64_t dcuyasheint_t;
#define BN_DIGIT WORD
typedef struct bn_st{
	int alloc;
	int used;
	int sign;
	cuyasheint_t *dp;
} bn_t;

__host__  void bn_new(bn_t *a);
__host__ __device__ void bn_zero(bn_t *a);
__host__ __device__ void bn_set_dig(bn_t *a, cuyasheint_t digit);
__host__  void bn_free(bn_t *a);
__host__ void bn_grow(bn_t *a,const unsigned int new_size);
__host__ void get_words(bn_t *b,ZZ a);
__host__ __device__ cuyasheint_t bn_mod1_low(const cuyasheint_t *a,
									const int size,
									const cuyasheint_t b);
__host__ __device__ cuyasheint_t bn_addn_low(cuyasheint_t *c,
									cuyasheint_t *a,
									cuyasheint_t *b,
									const int size
									);
__host__ __device__ cuyasheint_t bn_mul1_low(cuyasheint_t *c,
									const cuyasheint_t *a,
									cuyasheint_t digit,
									int size);
ZZ get_ZZ(bn_t *a);


void crt(bn_t *coefs,cuyasheint_t *d_polyCRT,const int N, const int NPolis,cudaStream_t stream);
void icrt(bn_t *coefs,cuyasheint_t *d_polyCRT,const int N, const int NPolis,cudaStream_t stream);


#endif
