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

def INTT(a):
	global WInv
	value = []
	for i in range(N):
		value.append((sum([x[0]*x[1] for x in zip(WInv[i],a)]))%P)
	return [x/N for x in value]


def mul(a,b):
	return [x[0]*x[1] % P for x in zip(a,b)]

def add(a,b):
	return [x[0]+x[1] % P for x in zip(a,b)]

a = range((N/2)) + [0]*(N/2)
b = [1]*(N/2) + [0]*(N/2)

assert INTT(NTT(a)) == a
assert INTT(NTT(b)) == b

from random import randint
for _ in range(100):
	a = [randint(pow(2,30),pow(2,31)) for _ in range(N)]
	assert INTT(NTT(a)) == a

A = NTT(a,W)
B = NTT(b,W)

C = mul(A,B)

c = INTT(C)
print c