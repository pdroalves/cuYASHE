#include "stockham_reference.h"
#include <math.h>


 __device__ float2 mulComplex(const float2 a,const float2 b){
	float2 aux;

	aux.x = a.x*b.x - a.y*b.y;
	aux.y = a.x*b.y + a.y*b.x;

	return aux;
}


__host__ __device__ void FFT(Complex *v){
	Complex v0 = v[0];
	#ifdef __CUDA_ARCH__
	v[0].x = v0.x + v[1].x;
	v[0].y = v0.y + v[1].y;
	v[1].x = v0.x - v[1].x;
	v[1].y = v0.y - v[1].y;
	#else
	v[0] = v0 + v[1];
	v[1] = v0 - v[1];
	#endif
}

__host__ __device__ int expand(int idxL, int N1, int N2){
	return (idxL/N1)*N1*N2 + (idxL%N1);
}

__device__ __host__ void FFTIteration(const int j,const int N,const int R,const int Ns, const Complex* data0, Complex *data1, const int type){
	// Complex *v = new Complex[R];
	Complex v[2];
	int idxS = j;
	float angle;
	if(type == FORWARD)
		angle = -2*M_PI*(j%Ns)/(Ns*R);
	else
		angle = 2*M_PI*(j%Ns)/(Ns*R);

	for(int r=0; r<R; r++){
		v[r] = data0[idxS+r*N/R];
		#ifdef __CUDA_ARCH__
		v[r] = mulComplex(v[r],float2{cosf(r*angle),sinf(r*angle)});
		#else
		v[r] *= Complex{cosf(r*angle),sinf(r*angle)};
		#endif
	}
	FFT(v);
	int idxD = expand(j,Ns,R);
	for(int r=0; r<R;r++)
		data1[idxD+r*Ns] = v[r];
}

__host__ Complex* CPU_FFT(int N,int R, Complex* data0, Complex* data1,const int type){
	for(int Ns=1; Ns<N; Ns*=R){
		for(int j=0; j < N/R; j++)
			FFTIteration(j, N, R, Ns, data0,data1,type);
		std::swap(data0,data1);
	}
	return data0;
}
__global__ void GPU_FFT(const int N, const int R, const int Ns, Complex* dataI, Complex* dataO,const int type){

	int j = (blockIdx.x)*N + (threadIdx.x);

	FFTIteration(j, N, R, Ns, dataI, dataO,type);
}

__host__ void CALL_GPU_FFT(int N,int R, void* data0, void* data1,const int type){
	dim3 blockDim(min(N/R,1024));
	dim3 gridDim((N/R)/blockDim.x);
	for(int Ns=1; Ns<N; Ns*=R){
		GPU_FFT<<<gridDim,blockDim >>>(N,R,Ns,(Complex*)data0,(Complex*)data1,type);
    	assert(cudaGetLastError() == cudaSuccess);
	}
}