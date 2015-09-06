#include "stockham_ntt_reference.h"
#include <math.h>

__device__ __host__ uint64_t s_rem (uint64_t a)
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

__device__ __host__  uint64_t s_mul(volatile uint64_t a,volatile uint64_t b){
  // Multiply and reduce a and b by prime 2^64-2^32+1
  const uint64_t P = 18446744069414584321;

  // Multiply
  #ifdef __CUDA_ARCH__
  uint64_t cHi = __umul64hi(a,b);
  uint64_t cLo = a*b;
 

  // Reduce
  uint64_t x3 = (cHi >> 32);
  uint64_t x2 = (cHi & UINT32_MAX);
  uint64_t x1 = (cLo >> 32);
  uint64_t x0 = (cLo & UINT32_MAX);

  uint64_t X1 = (x1<<32);
  uint64_t X2 = (x2<<32);

  
  uint64_t res = (X1+X2+x0-x2-x3);
  bool testA = (((x2+x3) > X1+X2+x0) && !((X1+X2 < X1) ||  ((X1+X2)+x0 < x0)));
  bool testB = ((res >= P) );
  bool testC = ((X1+X2 < X1) ||  ((X1+X2)+x0 < x0));

  // This avoids conditional branchs
  res = (P-res)*(testA) + (res-P)*(!testA && testB) + (res+4294967295L)*(!testA && !testB && testC) + (res)*(!testA && !testB && !testC);


  // if(((x2+x3) > X1+X2+x0) && !((X1+X2 < X1) ||  ((X1+X2)+x0 < x0))) {
  //   // printf("Negative Overflow!\n");
  //   res = P - res;
  //  }else if ((res >= P) ){
  //   // printf(" Overflow!\n");
  //   res -= P;
  //  }else if((X1+X2 < X1) ||  ((X1+X2)+x0 < x0)){
  //   // printf(" Double Overflow\n");
  //   res += 4294967295L;
  // }
   #else
  uint64_t res = (((__uint128_t)a) * ((__uint128_t)b) )%P; 
  // __uint128_t c = ((__uint128_t)a) * ((__uint128_t)b); 
  // uint64_t cHi = (c>>64);
  // uint64_t cLo = (c & UINT64_MAX);
  #endif
  return res;
}

__device__ __host__  uint64_t s_add(volatile uint64_t a,volatile uint64_t b){
  // Multiply and reduce a and b by prime 2^64-2^32+1
  uint64_t res = a+b;
  res += (res < a)*4294967295L;
  res = s_rem(res);
  return res;
}

__host__ __device__ void NTT(integer *v){
	// Butterfly
	integer v0 = v[0];
	v[0] = s_add(v0,v[1]);
	v[1] = v0 - v[1];
}

__host__ __device__ int expand(int idxL, int N1, int N2){
	return (idxL/N1)*N1*N2 + (idxL%N1);
}

__device__ __host__ void NTTIteration(	integer *W,
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
  #ifndef __CUDA_ARCH__
  if(idxD+0*Ns == 1)
    std::cout << idxS+0*N/R << " -> " << idxD+0*Ns << " = " << " (" << data0[idxS+0*N/R] << "*" << w[w_index] << ") = " << v[0] << std::endl;
  if(idxD+1*Ns == 1)
    std::cout << idxS+1*N/R << " -> " << idxD+1*Ns << " = " << " (" << data0[idxS+1*N/R] << "*" << w[w_index] << ") = " << v[1] << std::endl;
  #endif
  
  NTT(v);
	for(int r=0; r<R;r++)
		data1[idxD+r*Ns] = v[r];
  #ifndef __CUDA_ARCH__
  if(idxD+0*Ns == 1)
    std::cout << idxS+0*N/R << " = " << " 0) (" << data0[idxS+0*N/R] << "*" << w[w_index] << ") = " << v[0] << ", wN: "<< w_index << std::endl;
  if(idxD+1*Ns == 1)
    std::cout << idxS+1*N/R << " = " << " 1) (" << data0[idxS+1*N/R] << "*" << w[w_index] << ") = " << v[1] << ", wN: "<< w_index << std::endl;
  #endif

}

__host__ integer* CPU_NTT(integer *h_W,integer *h_WInv,int N,int R, integer* data0, integer* data1,const int type){
	for(int Ns=1; Ns < N; Ns*=R){
    std::cout << "Ns: " << Ns << std::endl;
		for(int j=0; j < N/R; j++)
			NTTIteration(h_W,h_WInv,j, N, R, Ns, data0,data1,type);
    printf("Swap!\n");
		// std::swap(data0,data1);
	}
	return data0;
}
__global__ void GPU_NTT(integer *d_W,integer *d_WInv,const int N, const int R, const int Ns, integer* dataI, integer* dataO,const int type){

  for(int i = 0; i < N/R; i += 1024){
    // " Threads virtuais "
    const int j = (blockIdx.x)*N + (threadIdx.x+i);
  	if( j < N)
	  	NTTIteration(d_W,d_WInv,j, N, R, Ns, dataI, dataO,type);
  }
}

__host__ void CALL_GPU_NTT(integer *d_W,integer *d_WInv,int N,int R, integer* data0, integer* data1,const int type){
	dim3 blockDim(min(N/R,1024));
	dim3 gridDim((N/R)/blockDim.x);
	for(int Ns=1; Ns<N; Ns*=R){
		GPU_NTT<<<gridDim,blockDim >>>(d_W,d_WInv,N,R,Ns,(integer*)data0,(integer*)data1,type);
    	assert(cudaGetLastError() == cudaSuccess);
    	std::swap(data0,data1);
	}
    // std::swap(data0,data1);
}
