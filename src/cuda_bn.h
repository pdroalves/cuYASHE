#ifndef CUDA_BN_H
#define CUDA_BN_H

#include <NTL/ZZ.h>
#include "settings.h"

NTL_CLIENT 
//////////////////////
// Big number's stuff //
//////////////////////

typedef uint64_t dcuyasheint_t;
#define BN_DIGIT WORD
typedef struct bn_st{
	int alloc;
	int used;
	int sign;
	cuyasheint_t *dp;

} bn_t;

#define BN_POS      (0)
#define BN_NEG      (1)

ZZ get_ZZ(bn_t a);
void get_words(bn_t b,ZZ a);
void icrt(bn_t *coefs,cuyasheint_t *d_polyCRT,const int N, const int NPolis);


#endif
