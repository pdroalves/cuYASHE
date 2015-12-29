#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H
#include <cuda_runtime.h>
#include <stdint.h>
#include <assert.h>
#include "settings.h"
#include <NTL/ZZ.h>
#ifdef CUFFTMUL
#include <cufft.h>
#endif
#include "cuda_bn.h"

NTL_CLIENT

// __constant__ int PrimesL;
#define MAX_PRIMES_ON_C_MEMORY 4096
// cuyasheint_t *CRTPrimesGlobal;

class CUDAFunctions{
  public:
  	static int N;
    static int std_bn_t_alloc;
    //////////////////////////
    // CRT global variables //
    //////////////////////////
    static bn_t M;
    static bn_t* Mpis;
    static cuyasheint_t *invMpis;

    #ifdef NTTMUL
    /////////
    // NTT //
    /////////
    static cuyasheint_t wN;
    static cuyasheint_t *d_W;
    static cuyasheint_t *d_WInv;
    #elif defined(CUFFTMUL)
    ///////////
    // cuFFT //
    ///////////
    static cufftHandle plan;
    #endif

    static cuyasheint_t* callPolynomialAddSub(cudaStream_t stream,
                                            cuyasheint_t *a,
                                            cuyasheint_t *b,
                                            int size,
                                            int OP);
    static void callPolynomialAddSubInPlace(cudaStream_t stream,
                                            cuyasheint_t *a,
                                            cuyasheint_t *b,
                                            int size,
                                            int OP);
    static cuyasheint_t* callPolynomialMul(cudaStream_t stream,
                                            cuyasheint_t *a,
                                            const bool realign_A,
                                            const int A_N,
                                            cuyasheint_t *b,
                                            const bool realign_B,
                                            const int B_N,
                                            const int N,
                                            const int NPolis);
    static cuyasheint_t* callPolynomialOPInteger( const int opcode,
                                            cudaStream_t stream,
                                            cuyasheint_t *a,
                                            cuyasheint_t *integer_array,
                                            const int N,
                                            const int NPolis);
    static cuyasheint_t* callRealignCRTResidues(cudaStream_t stream,
                                            int oldSpacing,
                                            int newSpacing,
                                            cuyasheint_t *array,
                                            int residuesSize,
                                            int residuesQty);
    static void callNTT(const int N, const int NPolis,cuyasheint_t* dataI, cuyasheint_t* dataO,const int type);

    static void init(int N);
    static void write_crt_primes();
  private:
};
#ifdef NTTMUL
__device__ __host__ inline uint64_t s_rem (uint64_t a);
__device__ __host__  uint64_t s_mul(uint64_t a,
                                    uint64_t b);
#endif

#endif

