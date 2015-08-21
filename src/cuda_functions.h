#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H
#include <cuda_runtime.h>
#include <stdint.h>
#include <assert.h>
#include <NTL/ZZ.h>

NTL_CLIENT
// void host_bitreverse(dim3 gridDim,dim3 blockDim,long *a,int n,int npolys);
// void host_NTT(dim3 gridDim,dim3 blockDim,long *W,long *a, long *a_hat, long q,int N,int NPolis);

// __device__ __host__ uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
//     uint64_t res = 0;
//     uint64_t temp_b;

//     /* Only needed if b may be >= m */
//     if (b >= m) {
//         if (m > UINT64_MAX / 2u)
//             b -= m;
//         else
//             b %= m;
//     }

//     while (a != 0) {
//         if (a & 1) {
//             /* Add b to res, modulo m, without overflow */
//             if (b >= m - res) /* Equiv to if (res + b >= m), without overflow */
//                 res -= m;
//             res += b;
//         }
//         a >>= 1;

//         /* Double b, modulo m */
//         temp_b = b;
//         if (b >= m - b)        Equiv to if (2 * b >= m), without overflow 
//             temp_b -= m;
//         b += temp_b;
//     }
//     return res;
// }


// uint64_t powerMod (uint64_t b, uint64_t e, uint64_t mod)
// {
//     uint64_t result = 1;
//     unsigned significant = 1;
//     {
//         uint64_t e_t = e;

//         while (e_t >>= 1)
//         {
//             ++significant;
//         }
//     }

//     for (int pos = significant-1; pos >= 0; --pos)
//     {
//         bool bit = e & (1 << pos);
//         result = mulmod(result, result, mod);

//         if (bit)
//             result = mulmod(result,b, mod);
//     }

//     return result;
// }

// uint64_t invMod(uint64_t x,uint64_t p){
//   return (powerMod(x,p-2,p));

// }

class CUDAFunctions{
  public:
  	static uint64_t wN;
	static uint64_t *d_W;
	static uint64_t *d_WInv;
	const static uint64_t P = 18446744069414584321;//31 bits

    static uint32_t* callPolynomialAddSub(cudaStream_t stream,uint32_t *a,uint32_t *b,int size,int OP);
    static uint32_t* callPolynomialMul(cudaStream_t stream,uint32_t *a,uint32_t *b, int N, int NPolis);
    static uint32_t* callRealignCRTResidues(cudaStream_t stream,int oldSpacing,int newSpacing, uint32_t *array,int residuesSize,int residuesQty);
    static void init(int N){

			uint64_t *h_W;
			uint64_t *h_WInv;

			// const uint64_t P = 4294955009;//31 bits
			ZZ PZZ = conv<ZZ>("18446744069414584321");
			uint64_t k = conv<uint64_t>(PZZ-1)/N;
			ZZ wNZZ = NTL::PowerMod(ZZ(7),k,PZZ);
			// assert((P-1)%(N) == 0);
			// const uint64_t k = (P-1)/N;
			wN = conv<uint64_t>(wNZZ);
			// wN = 17870292113338400769;

		    h_W = (uint64_t*)malloc(N*N*sizeof(uint64_t));
			cudaError_t result = cudaMalloc((void**)&d_W,N*N*sizeof(uint64_t));
			assert(result == cudaSuccess);
			h_WInv = (uint64_t*)malloc(N*N*sizeof(uint64_t));
			result = cudaMalloc((void**)&d_WInv,N*N*sizeof(uint64_t));
			assert(result == cudaSuccess);

			  std::cout << "wN == " << wN << std::endl;
			  std::cout << "k == " << k << std::endl;
			  std::cout << "N == " << N << std::endl;
			  std::cout << "P == " << P << std::endl;
  
			// Computes W
			for(int j = 0; j < N; j++)
				for(int i = 0; i < N; i++)
				    // h_W[i+j*N] = (( j == 0)? 1:(h_W[i-1+j*N]*pow(wN,i)%q));
				    h_W[i+j*N] = conv<uint64_t>(NTL::PowerMod(wNZZ,j*i,PZZ));

			for(int j = 0; j < N; j++)
				for(int i = 0; i < N; i++)
				    h_WInv[i+j*N] = conv<uint64_t>(NTL::InvMod(conv<ZZ>(h_W[i+j*N]),PZZ ));

			result = cudaMemcpy(d_W,h_W , N*N*sizeof(uint64_t), cudaMemcpyHostToDevice);
			assert(result == cudaSuccess);
			result = cudaMemcpy(d_WInv,h_WInv , N*N*sizeof(uint64_t), cudaMemcpyHostToDevice);
			assert(result == cudaSuccess);

			free(h_W);
			free(h_WInv);
    }
  private:
};

#endif
