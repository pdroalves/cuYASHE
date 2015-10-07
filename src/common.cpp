#include <stdio.h>

#include "cuda_functions.h"
#include "polynomial.h"
#include "ciphertext.h"


template Polynomial common_addition<Polynomial>(Polynomial *a,Polynomial *b);
template Ciphertext common_addition<Ciphertext>(Ciphertext *a,Ciphertext *b);