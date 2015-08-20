wN = 4270698452L
P = 4294955009
N = 128

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

a = range(64) + [0]*64
b = [1]*64 + [0]*64

def NTT(a,W):
	value = []
	for i in range(N):
		value.append((sum([x[0]*x[1] for x in zip(W[i],a)]))%P)
	return value

def mul(a,b):
	return [x[0]*x[1] for x in zip(a,b)]

assert [x/N for x in NTT(NTT(a,W),WInv)] == a
assert [x/N for x in NTT(NTT(b,W),WInv)] == b


A = NTT(a,W)
B = NTT(b,W)

C = mul(A,B)

c = [x/N for x in NTT(C,WInv)]
print c