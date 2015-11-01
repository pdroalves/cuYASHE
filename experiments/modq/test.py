import crt
from polynomial import Polynomial
from random import random

q = 987587
M,primes = crt.get_primes(8,q,10)
P = Polynomial()
P.coef = [int(q*random()) for i in xrange(8)]

residues = crt.crt(P,primes)

assert crt.icrt(residues,M,primes) == P

residuesP2 = [x*x for x in residues]

assert crt.icrt(residuesP2) == (P*P % q)  