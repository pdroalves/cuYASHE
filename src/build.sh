#!/bin/bash

TOBUILD=(polynomial.o cuda_functions.o cuda_bn.o cuda_distribution.o cuda_ciphertext.o test.o test_ctpt.o test_ntt.o test_mul.o test_crt.o test_reduce.o test_bnt.o ciphertext.o distribution.o yashe.o common.o)

for i in "${TOBUILD[@]}"
do
	make $i &
done

wait

make
