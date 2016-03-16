#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H
#include <cuda_runtime.h>
#include <stdint.h>
#include <assert.h>
#include "settings.h"
#include <NTL/ZZ.h>
// #ifdef CUFFTMUL
#include <cufft.h>
// #endif
#include "cuda_bn.h"

NTL_CLIENT

// __constant__ int PrimesL;
#define MAX_PRIMES_ON_C_MEMORY 4096
// cuyasheint_t *CRTPrimesGlobal;
typedef double2 Complex;

class CUDAFunctions{
  public:
  	static int N;
    static int std_bn_t_alloc;
    static int transform;
    //////////////////////////
    // CRT global variables //
    //////////////////////////
    static cuyasheint_t *d_inner_results;
    static cuyasheint_t *d_inner_results_used;

    /////////
    // NTT //
    /////////
    static cuyasheint_t wN;
    static cuyasheint_t *d_W;
    static cuyasheint_t *d_WInv;
    static cuyasheint_t *d_mulA;
    static cuyasheint_t *d_mulB;
    static cuyasheint_t *d_mulAux;
    ///////////
    ///////////
    // cuFFT //
    ///////////
    static cufftHandle plan;
    static Complex *d_mulComplexA;
    static Complex *d_mulComplexB;
    static Complex *d_mulComplexC;
    static cuyasheint_t* applyNTT(  cuyasheint_t *d_a,
                                    const int N,
                                    const int NPolis,
                                    int type,
                                    cudaStream_t stream);
    static void callPolynomialAddSub(cuyasheint_t *c,
                                            cuyasheint_t *a,
                                            cuyasheint_t *b,
                                            int size,
                                            int OP,
                                            cudaStream_t stream);
    static void callPolynomialAddSubInPlace(cudaStream_t stream,
                                            cuyasheint_t *a,
                                            cuyasheint_t *b,
                                            int size,
                                            int OP);

    static void executeNTTScale(    cuyasheint_t *a, 
                                    const int size, 
                                    const int N,
                                    cudaStream_t stream);
    static void executePolynomialMul(cuyasheint_t *c, 
                                    cuyasheint_t *a, 
                                    cuyasheint_t *b, 
                                    const int size, 
                                    cudaStream_t stream);
    static void executeCuFFTPolynomialMul( Complex *a, 
                                                    Complex *b, 
                                                    Complex *c, 
                                                    int size_c, 
                                                    int size, 
                                                    cudaStream_t stream);
    static void executePolynomialAdd(cuyasheint_t *c, 
                                    cuyasheint_t *a, 
                                    cuyasheint_t *b, 
                                    const int size, 
                                    cudaStream_t stream);
    static void executeCopyIntegerToComplex(   Complex *d_a, 
                                                            cuyasheint_t *a,
                                                            const int size,
                                                            cudaStream_t stream);
    static void executeCopyAndNormalizeComplexRealPartToInteger(   cuyasheint_t *d_a, 
                                                                                cufftDoubleComplex *a,
                                                                                const int size,
                                                                                int signal_size,
                                                                                int N,
                                                                                cudaStream_t stream);
    static cuyasheint_t* callPolynomialMul( cuyasheint_t *d_result,
                                            cuyasheint_t *a,
                                            const bool realign_A,
                                            const int A_N,
                                            cuyasheint_t *b,
                                            const bool realign_B,
                                            const int B_N,
                                            const int N,
                                            const int NPolis,
                                            cudaStream_t stream);
    static cuyasheint_t* callPolynomialOPInteger( const int opcode,
                                            cudaStream_t stream,
                                            cuyasheint_t *a,
                                            cuyasheint_t *integer_array,
                                            const int N,
                                            const int NPolis);
    static void callPolynomialOPDigit( const int opcode,
                                            cudaStream_t stream,
                                            bn_t *a,
                                            bn_t digit,
                                            const int N);
    static cuyasheint_t* callRealignCRTResidues(cudaStream_t stream,
                                            int oldSpacing,
                                            int newSpacing,
                                            cuyasheint_t *array,
                                            int residuesSize,
                                            int residuesQty);
    static void callNTT(const int N, const int NPolis,int RADIX, cuyasheint_t* dataI, cuyasheint_t* dataO,const int type);
    static void init(int N);
    static void write_crt_primes();
  private:
};
__device__ __host__ inline uint64_t s_rem (uint64_t a);
__device__ __host__  uint64_t s_mul(uint64_t a,
                                    uint64_t b);
#endif