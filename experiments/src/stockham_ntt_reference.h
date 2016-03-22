#ifndef STOCKHAM_REFERENCE_H
#define STOCKHAM_REFERENCE_H
#include <assert.h>
#include <iostream>

#define RADIX 2
#define BILLION  1000000000L
#define MILLION  1000000L
#define NITERATIONS 100
// #define FIRSTITERATION 128
// #define LASTITERATION 524288
#define FIRSTITERATION 32
#define LASTITERATION 32

typedef uint64_t integer;
enum {FORWARD,INVERSE};

integer* CPU_NTT(integer *h_W,integer *h_WInv,int N,int R, integer* data0, integer* data1,const int type);
void CALL_GPU_NTT(integer *d_W,integer *d_WInv,int N,int R, integer* data0, integer* data1,const int type);

#endif 