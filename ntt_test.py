# P = 4294955009
P = pow(2,64)-pow(2,32)+1
# P = pow(2,32)-pow(2,20)+1
N = 32
assert( (P-1) % N == 0)
# wN = pow(7,(P-1)/N,P)
wN = pow(7,(P-1)/N,P)

W = []
for i in range(N):
	W.append([])
	for j in range(N):
		W[i].append(pow(wN,j*i,P))


WInv = []
for i in range(N):
	WInv.append([])
	for j in range(N):
		WInv[i].append(pow(W[i][j],P-2,P))


def NTT(a):
	global W
	value = []
	for i in range(N):
		value.append((sum([x[0]*x[1] for x in zip(W[i],a)]))%P)
	return value

def normalize(a):
	A = list(a)
	while A[-1] == 0:
		A.pop()
	return A

def INTT(a):
	global WInv
	value = []
	for i in range(N):
		value.append((sum([x[0]*x[1] for x in zip(WInv[i],a)]))%P)
	return normalize([x/N for x in value])

def mul(a,b):
	return [x[0]*x[1] % P for x in zip(a,b)]

def add(a,b):
	return [(x[0]+x[1]) % P for x in zip(a,b)]

def cmp(a,b):
	A = normalize(a)
	B = normalize(b)

	return A == B

def fix(c):
	x  = Polynomial()
	x.coef = c
	return (x%phi%q).coef

a = range((N/2)) + [0]*(N/2)
b = [1]*(N/2) + [0]*(N/2)

assert cmp(INTT(NTT(a)), a)
assert cmp(INTT(NTT(b)), b)

from random import randint
for _ in range(100):
	a = [randint(pow(2,30),pow(2,31)) for _ in range(N)]
	assert cmp(INTT(NTT(a)), a)

phi = Polynomial()
phi.set_dth_cyclotomic(N)
q = 2**127-1

a = Polynomial()
a.coef = [randint(pow(2,9),pow(2,10)) for _ in range(N/2)] + [0]*(N/2)
b = Polynomial()
b.coef = [randint(pow(2,9),pow(2,10)) for _ in range(N/2)] + [0]*(N/2)

A = NTT(a)
B = NTT(b)

C = mul(A,B)
for i in range(3431):
	print i
	C = add(C,A)
	C = add(C,B)
	assert INTT(C) == (a*b) + (a+b)*(i+1) % P