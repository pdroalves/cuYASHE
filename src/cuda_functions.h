#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H
#include <cuda_runtime.h>

class CUDAFunctions{
  public:
    static long* callPolynomialAddSub(cudaStream_t stream,long *a,long *b,int size,int OP);
    static long* callRealignCRTResidues(cudaStream_t stream,int oldSpacing,int newSpacing, long *array,int residuesSize,int residuesQty);
  private:
};

#endif
