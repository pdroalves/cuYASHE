#include "stockham_reference.h"
#include <math.h>

void FFT(Complex *v){
	Complex v0 = v[0];
	v[0] = v0 + v[1];
	v[1] = v0 - v[1];
}

int expand(int idxL, int N1, int N2){
	return (idxL/N1)*N1*N2 + (idxL%N1);
}

void FFTIteration(	const int j,
					const int N,
					const int R,
					const int Ns,
					const Complex* data0,
					Complex *data1,
					const int type
				){
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
		v[r] *= Complex{cosf(r*angle),sinf(r*angle)};
	}
	FFT(v);
	int idxD = expand(j,Ns,R);
	for(int r=0; r<R;r++){
		data1[idxD+r*Ns] = v[r];
	}
}

Complex* CPU_FFT(int N,int R, Complex* data0, Complex* data1,const int type){
	for(int Ns=1; Ns<N; Ns*=R){
		for(int j=0; j < N/R; j++)
			FFTIteration(j, N, R, Ns, data0,data1,type);
		std::swap(data0,data1);
	}
	return data0;
}