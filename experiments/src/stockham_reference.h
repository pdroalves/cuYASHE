#ifndef STOCKHAM_REFERENCE_H
#define STOCKHAM_REFERENCE_H
#include <assert.h>
#include <iostream>
#include <cuda_runtime_api.h>

#define RADIX 2
#define BILLION  1000000000L
#define MILLION  1000000L
#define NITERATIONS 100
#define FIRSTITERATION 1024
#define LASTITERATION 8192

// #define FIRSTITERATION 256
// #define LASTITERATION 256

typedef struct complex{
	float real;
	float imag;


	__host__ __device__  complex operator+(const complex b){
		complex aux;
		aux.real = this->real + b.real;
		aux.imag = this->imag + b.imag;
		return aux;
	}

	__host__ __device__  complex operator-(const complex b){
		complex aux;
		aux.real = this->real - b.real;
		aux.imag = this->imag - b.imag;
		return aux;
	}
	__host__ __device__  complex operator/(const float b){
		complex aux;
		aux.real = this->real / b;
		aux.imag = this->imag / b;
		return aux;

	}
	// complex operator*=(const complex b){
	// 	//this has better precision
	// 	complex aux;
		
	// 	complex cauxA(this->real,this->imag);
	// 	complex cauxB(b.real,b.imag);
	// 	aux.real = (cauxA * cauxB).real();
	// 	aux.imag = (cauxA * cauxB).imag();

	// 	return aux;
	// }
	__host__ __device__ complex operator*=(const complex b){
		complex aux;

		aux.real = this->real*b.real - this->imag*b.imag;
		aux.imag = this->real*b.imag + this->imag*b.real;
		this->real = aux.real;
		this->imag = aux.imag;

		return *this;

	}
	__host__ __device__ complex operator/=(const float b){
		this->real /= b;
		this->imag /= b;
		return *this;

	}
} complex;


#ifdef __CUDA_ARCH__
typedef float2 Complex;
#else
typedef complex Complex ;
#endif


enum {FORWARD,INVERSE};

complex* CPU_FFT(int N,int R, complex* data0, complex* data1,const int type);
void CALL_GPU_FFT(int N,int R, void* data0, void* data1,const int type);

#endif 