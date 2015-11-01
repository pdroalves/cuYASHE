#!/usr/bin/python
import generate_prime as Prime
from polynomial import Polynomial

def get_primes(n,q,primes_size):
	# generate primes for polynomials os degree n with coefficients mod q
	M = 1
	primes = list()
	while M < n*q*q:
		pi = Prime.generate_large_prime(primes_size)
		if pi not in primes:
			M = M*pi
			primes.append(pi)
	return M,primes

def crt(p,primes):
	result = []
	for pi in primes:
		result.append(p % pi)
	return result

def icrt(p,M,primes):
	result = Polynomial()
	for i,pi in enumerate(primes):
		Mpi = M/pi
		invMpi = pow(Mpi,pi-2,pi)
		result = result + (Mpi*(invMpi*p[i] % pi))
	result = result % M
	return result

