#include "stockham_ntt_reference.h"
#include <math.h>

bool overflow(const uint64_t a, const uint64_t b){
  // True if a+b will result in a integer overflow.
  return (a+b) < a;
  // return lessThan((a+b),a);
}

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

uint64_t mulhi(uint64_t a,uint64_t b){
uint64_t    a_lo = (uint32_t)a;
uint64_t    a_hi = a >> 32;
uint64_t    b_lo = (uint32_t)b;
uint64_t    b_hi = b >> 32;

uint64_t    a_x_b_hi =  a_hi * b_hi;
uint64_t    a_x_b_mid = a_hi * b_lo;
uint64_t    b_x_a_mid = b_hi * a_lo;
uint64_t    a_x_b_lo =  a_lo * b_lo;

uint64_t    carry_bit = ((uint64_t)(uint32_t)a_x_b_mid +
                         (uint64_t)(uint32_t)b_x_a_mid +
                         (a_x_b_lo >> 32) ) >> 32;

uint64_t    multhi = a_x_b_hi +
                     (a_x_b_mid >> 32) + (b_x_a_mid >> 32) +
                     carry_bit;

return multhi;
}

uint64_t s_mul(uint64_t a,uint64_t b){
  // Multiply and reduce a and b by prime 2^64-2^32+1
  //const uint64_t P = 18446744069414584321;
  const uint64_t PRIMEP = 18446744069414584321;

  // Multiply
  //uint64_t res = (((__uint128_t)a) * ((__uint128_t)b) )%P; 
  

  const uint64_t GAP = (UINT64_MAX-PRIMEP+1);

  const uint64_t cHi = mulhi(a,b);
  const uint64_t cLo = a*b;


  // Reduce
  const uint64_t x3 = (cHi >> 32);
  const uint64_t x2 = (cHi & UINT32_MAX);
  const uint64_t x1 = (cLo >> 32);
  const uint64_t x0 = (cLo & UINT32_MAX);

  const uint64_t X1 = (x1<<32);
  const uint64_t X2 = (x2<<32);

  ///////////////////////////////
  //
  // Here we can see three kinds of overflows:
  //
  // * Negative overflow: Result is negative. 
  // Since uint64_t uses mod UINT64_MAX, we need to translate to the correct value mod PRIMEP.
  // * Simple overflow: Result is bigger than PRIMEP but not enough to exceed UINT64_MAX.
  //  We solve this in the same way we solve negative overflow, just translate to the correct value mod PRIMEP.
  // * Double overflow

  uint64_t res = X1+X2+x0-x2-x3;
  const bool testA = (x2+x3 > X1+X2+x0) && !( overflow(X1,X2) ||  overflow(X1+X2,x0) ); // Negative overflow
  const bool testB = ( res >= PRIMEP ); // Simple overflow
  const bool testC = (overflow(X1,X2) || overflow(X1+X2,x0)) && (X1+X2+x0 > x2+x3); // Double overflow

  // This avoids conditional branchs
  // res = (PRIMEP-res)*(testA) + (res-PRIMEP)*(!testA && testB) + (PRIMEP - (UINT64_MAX-res))*(!testA && !testB && testC) + (res)*(!testA && !testB && !testC);
  res =   (PRIMEP-res)*(testA) 
        + (res-PRIMEP)*(!testA && testB) 
        + (res+GAP)*(!testA && !testB && testC) 
        + (res)*(!testA && !testB && !testC);


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
