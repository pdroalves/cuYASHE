#include "stockham_ntt_reference.h"
#include <math.h>

__device__ __host__ inline integer s_rem (uint64_t a)
{
  // P = pow(2,31)-1
  uint64_t res = (a>>31) + (a&0x7FFFFFFF);
  // This implies in an annoying divergence
  if(res > 0x7FFFFFFF)
    res = (integer)((res>>31) + (res&0x7FFFFFFF));
  #ifdef __CUDA_ARCH__
  // Only for device code
  __syncthreads();
  #endif
  return (integer)res;
}

__device__ __host__ inline uint64_t s_mul(uint64_t a,uint64_t b){
	// Multiply and reduce a and b by prime 2^64-2^32+1

	// Multiply
	#ifdef __CUDA_ARCH__
	uint64_t cHi = __umul64hi(a,b);
	uint64_t cLo = a*b;
	#else
	__uint128_t c = ((__uint128_t)a) * ((__uint128_t)b); 
	uint64_t cHi = (c>>64);
	uint64_t cLo = (c & UINT64_MAX);
	#endif

	// Reduce
	uint64_t x3 = (cHi >> 32);
	uint64_t x2 = (cHi & UINT32_MAX);
	uint64_t x1 = (cLo >> 32);
	uint64_t x0 = (cLo & UINT32_MAX);

	uint64_t res = ((x1+x2)<<32)+x0-x2-x3;
	return res - (0xffffffff00000001)*(res >= 0xffffffff00000001);
}

__host__ __device__ void NTT(integer *v){
	// Butterfly
	integer v0 = v[0];
	v[0] = v0 + v[1];
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
	// int wIndex;
	integer *w;
	if(type == FORWARD)
		w = W;
	else
		w = WInv;


	for(int r=0; r<R; r++)
		// v[r] = s_rem(data0[idxS+r*N/R]*w[j]);
		v[r] = s_mul(data0[idxS+r*N/R],w[j]);
	
	NTT(v);
	const int idxD = expand(j,Ns,R);
	for(int r=0; r<R;r++)
		data1[idxD+r*Ns] = v[r];
}

__host__ integer* CPU_NTT(integer *h_W,integer *h_WInv,int N,int R, integer* data0, integer* data1,const int type){
	for(int Ns=1; Ns<N; Ns*=R){
		for(int j=0; j < N/R; j++)
			NTTIteration(h_W,h_WInv,j, N, R, Ns, data0,data1,type);
		std::swap(data0,data1);
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
	}
}
