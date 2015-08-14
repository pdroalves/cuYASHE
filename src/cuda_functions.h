#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H
#include <cuda_runtime.h>

void host_bitreverse(dim3 gridDim,dim3 blockDim,long *a,int n,int npolys);
void host_NTT(dim3 gridDim,dim3 blockDim,long *W,long *a, long *a_hat, long q,int N,int NPolis);

class CUDAFunctions{
  public:
    static long* callPolynomialAddSub(cudaStream_t stream,long *a,long *b,int size,int OP);
    static long* callPolynomialMul(cudaStream_t stream,long *a,long *b, int N, int NPolis);
    static long* callRealignCRTResidues(cudaStream_t stream,int oldSpacing,int newSpacing, long *array,int residuesSize,int residuesQty);
  private:
};

#endif
