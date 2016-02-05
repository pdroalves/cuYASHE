# P = 4294955009
P = pow(2,64)-pow(2,32)+1
# P = pow(2,32)-pow(2,20)+1
N = 16
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


def NTT(a,W):
	value = []
	for i in range(N):
		value.append((sum([x[0]*x[1] for x in zip(W[i],a)]))%P)
	return value

def mul(a,b):
	return [x[0]*x[1] % P for x in zip(a,b)]

a = range((N/2)) + [0]*(N/2)
b = [1]*(N/2) + [0]*(N/2)

assert [x/N for x in NTT(NTT(a,W),WInv)] == a
assert [x/N for x in NTT(NTT(b,W),WInv)] == b

from random import randint
for _ in range(100):
	a = [randint(pow(2,30),pow(2,31)) for _ in range(N)]
	assert [x/N for x in NTT(NTT(a,W),WInv)] == a

A = NTT(a,W)
B = NTT(b,W)

C = mul(A,B)

c = [x/N for x in NTT(C,WInv)]
print c