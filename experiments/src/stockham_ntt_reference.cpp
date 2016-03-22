#include "stockham_ntt_reference.h"
#include <math.h>

uint64_t s_rem (uint64_t a)
{
  // Special reduction for prime 2^64-2^32+1
  //
  // x3 * 2^96 + x2 * 2^64 + x1 * 2^32 + x0 \equiv
  // (x1+x2) * 2^32 + x0 - x3 - x2 mod p
  //
  // Here: x3 = 0, x2 = 0, x1 = (a >> 32), x0 = a-(x1 << 32)
  // const uint64_t p = 0xffffffff00000001;
  // uint64_t x3 = 0;
  // uint64_t x2 = 0;
  uint64_t x1 = (a >> 32);
  uint64_t x0 = (a & UINT32_MAX);

  // uint64_t res = ((x1+x2)<<32 + x0-x3-x2);
  uint64_t res = ((x1<<32) + x0);

  return res;
} 

uint64_t s_mul(volatile uint64_t a,volatile uint64_t b){
  // Multiply and reduce a and b by prime 2^64-2^32+1
  const uint64_t P = 18446744069414584321;

  // Multiply
  uint64_t res = (((__uint128_t)a) * ((__uint128_t)b) )%P; 
  return res;
}

 uint64_t s_add(volatile uint64_t a,volatile uint64_t b){
  // Multiply and reduce a and b by prime 2^64-2^32+1
  uint64_t res = a+b;
  res += (res < a)*4294967295L;
  res = s_rem(res);
  return res;
}

void NTT(integer *v){
	// Butterfly
	integer v0 = v[0];
	v[0] = s_add(v0,v[1]);
	v[1] = v0 - v[1];
}

int expand(int idxL, int N1, int N2){
	return (idxL/N1)*N1*N2 + (idxL%N1);
}

void NTTIteration(	integer *W,
										integer *WInv,
										const int j,
										const int N,
										const int R,
										const int Ns,
										const integer* data0,
										integer *data1, 
										const int type){
	integer v[2];
	const int idxS = j;
	integer *w;
	if(type == FORWARD)
		w = W;
	else
		w = WInv;
  // const int w_index = idxD;
  const int w_index = ((j%Ns)*N)/(Ns*R);
  assert( ((j%Ns)*N)%(Ns*R) == 0);

  for(int r=0; r<R; r++){

    v[r] = s_mul(data0[idxS+r*N/R],w[w_index]);
  }
  const int idxD = expand(j,Ns,R);
  
  NTT(v);
	for(int r=0; r<R;r++)
		data1[idxD+r*Ns] = v[r];
}

integer* CPU_NTT(integer *h_W,integer *h_WInv,int N,int R, integer* data0, integer* data1,const int type){
	for(int Ns=1; Ns < N; Ns*=R){
		for(int j=0; j < N/R; j++)
			NTTIteration(h_W,h_WInv,j, N, R, Ns, data0,data1,type);
	}
	return data0;
}