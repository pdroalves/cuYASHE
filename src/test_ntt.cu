#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <iomanip>
#include <NTL/ZZ.h>
 #include <stdint.h>

#include "common.h"
#include "cuda_functions.h"

NTL_CLIENT

#define BILLION  1000000000L
#define MILLION  1000000L
#define DEGREE 64
#define NPOLYS 1
// #define P 0xffffffff00000001

double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}


// __device__ __host__ uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
//     int64_t res = 0;
//     while (a != 0) {
//         if (a & 1) res = (res + b) % m;
//         a >>= 1;
//         b = (b << 1) % m;
//     }
//     return res;
// }

__device__ __host__ uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t res = 0;
    uint64_t temp_b;

    /* Only needed if b may be >= m */
    if (b >= m) {
        if (m > UINT64_MAX / 2u)
            b -= m;
        else
            b %= m;
    }

    while (a != 0) {
        if (a & 1) {
            /* Add b to res, modulo m, without overflow */
            if (b >= m - res) /* Equiv to if (res + b >= m), without overflow */
                res -= m;
            res += b;
        }
        a >>= 1;

        /* Double b, modulo m */
        temp_b = b;
        if (b >= m - b)       /* Equiv to if (2 * b >= m), without overflow */
            temp_b -= m;
        b += temp_b;
    }
    return res;
}

__global__ void NTT(uint64_t *W,uint64_t *a, uint64_t *a_hat, int N,int NPolis,uint64_t P){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int cid = tid & (N-1); // Coefficient id

  // const uint64_t p = 0xffffffff00000001;
  if(tid < N*NPolis){
    uint64_t value = 0;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      value = (value + mulmod(W[i + cid*N],a[i],P)) %P;
    }
    a_hat[cid] = value % P;
  }

}



__global__ void DOUBLENTT2(uint64_t *W,uint64_t *a, uint64_t *a_hat,uint64_t *b, uint64_t *b_hat, int N,int NPolis,uint64_t P){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int cid = tid & (N-1); // Coefficient id

  // const uint64_t p = 0xffffffff00000001;
  if(tid < N*NPolis){
    uint64_t Wvalue;
    uint64_t Avalue;
    uint64_t Bvalue;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      Wvalue = W[i + cid*N];
      Avalue = (Avalue + Wvalue*a[i]) ;
      Bvalue = (Bvalue + Wvalue*b[i]) ;
    }
    a_hat[cid] = Avalue;
    b_hat[cid] = Bvalue;
  }

}


uint64_t SQM_pow (uint64_t b, uint64_t e, uint64_t mod)
{
    uint64_t result = 1;
    unsigned significant = 1;
    {
        uint64_t e_t = e;

        while (e_t >>= 1)
        {
            ++significant;
        }
    }

    for (int pos = significant-1; pos >= 0; --pos)
    {
        bool bit = e & (1 << pos);
        result = mulmod(result, result, mod);

        if (bit)
            result = mulmod(result,b, mod);
    }

    return result;
}

// ZZ ZZFromUint64 (uint64_t value)
// {
//     unsigned int unit = 256;
//     ZZ power;
//     power = 1;
//     ZZ res;
//     res = 0;

//     while (value > 0) {
//         res = res + ((long int)value % unit) * power;
//         power = power * unit;
//         value = value / unit;
//     }
//     return res;

// };
// uint64_t
// uint64FromZZ(ZZ val)
// {
//     uint64_t res = 0;
//     uint64_t mul = 1;
//     while (val > 0) {
//         res = res + mul*(to_int(val % 10));
//         mul = mul * 10;
//         val = val / 10;
//     }
//     return res;
// }

uint64_t powerMod(uint64_t x,uint64_t h,uint64_t p){
  // unsigned long t;
  // if(h == 0)
  //   return 1;
  // else if(h == 1)
  //   return x % p;
  // else
  //   t = log2((double)(h))+1;
  // ZZ r = ZZ(x);
  // ZZ X = ZZ(x);
  // ZZ P = ZZ(p);

  // for(int i = t-1; i >= 0; i--){
  //   r = r*r;
  //   r %= P;
  //   if((h >> i) & 1 == 1)//i-th bit
  //     r *= X % P;
    
  // }
  // return conv<uint64_t>(r);

  // return uint64FromZZ(NTL::PowerMod(ZZFromUint64(x),h,ZZFromUint64(p)));
  return (SQM_pow(x,h,p));
}

uint64_t invMod(uint64_t x,uint64_t p){
  // if(x == 0){
    // std:: cout << "Achei o erro!" << std::endl;
  // }
  // std::cout << x << std::endl;
  // return uint64FromZZ(NTL::InvMod(ZZFromUint64(x),ZZFromUint64(p)));
  // return uint64FromZZ(NTL::PowerMod(ZZFromUint64(x),p-2,ZZFromUint64(p)));

  return (SQM_pow(x,p-2,p));

}

// uint64_t hi(uint64_t x) {
//     return x >> 32;
// }

// uint64_t lo(uint64_t x) {
//     return ((1 << 32) - 1) & x;
// }

// void multiply(uint64_t a, uint64_t b) {
//     // actually uint32_t would do, but the casting is annoying
//     uint64_t s0, s1, s2, s3; 

//     uint64_t x = lo(a) * lo(b);
//     s0 = lo(x);

//     x = hi(a) * lo(b) + hi(x);
//     s1 = lo(x);
//     s2 = hi(x);

//     x = s1 + lo(a) * hi(b);
//     s1 = lo(x);

//     x = s2 + hi(a) * hi(b) + hi(x);
//     s2 = lo(x);
//     s3 = hi(x);

//     uint64_t result = s1 << 32 | s0;
//     uint64_t carry = s3 << 32 | s2;
// }



int main(void){

  // std:: cout <<  mulmod(6406262673276882058,4,9223372036854829057) << std::endl;
  // std:: cout <<  NTL::MulMod(6406262673276882058,4,9223372036854829057) << std::endl;

// uint64_t integer = 9223372036854829057L;
//     ZZ P = ZZ(integer);
//     ZZ x = Z(6209464568650184525);
//     ZZ inv = NTL::InvMod(x,P);
    
//     cout << integer << "\n" << P << endl;
    
//     cout << "PowerMod: " << inv << endl;
    
//     cout << "Check: " << NTL::MulMod(inv, Z(6209464568650184525), P) << endl;    
//   return 0;
  const int N = DEGREE;
  // const uint64_t P = 0xffffffff00000001;
  const uint64_t P = 9223372036854829057;//63 bits
  assert((P-1)%(N) == 0);
  const uint64_t k = (P-1)/N;
  // const uint64_t wN = powerMod(5,k,P);
  // const uint64_t wN = 549755813888;// Hard coded
  const uint64_t wN = 6209464568650184525;// Hard coded
  std::cout << "wN == " << wN << std::endl;
  std::cout << "k == " << k << std::endl;
  std::cout << "N == " << N << std::endl;
  std::cout << "P == " << P << std::endl;
  // std::cout << "prime == " << prime << std::endl;
  // const uint64_t q = 97;

	dim3 blockDim(ADDBLOCKXDIM);
	dim3 gridDim((N*NPOLYS)/ADDBLOCKXDIM+1);

	uint64_t *h_a;
	uint64_t *d_a;
	uint64_t *h_b;
  uint64_t *d_b;
  uint64_t *h_W;
  uint64_t *d_W;
  uint64_t *h_WInv;
  uint64_t *d_WInv;

	// Alloc memory
	h_a = (uint64_t*)malloc(N*NPOLYS*sizeof(uint64_t));
	h_b = (uint64_t*)malloc(N*NPOLYS*sizeof(uint64_t));
  cudaError_t result = cudaMalloc((void**)&d_a,N*NPOLYS*sizeof(uint64_t));
	assert(result == cudaSuccess);
  result = cudaMalloc((void**)&d_b,N*NPOLYS*sizeof(uint64_t));
	assert(result == cudaSuccess);

  h_W = (uint64_t*)malloc(N*N*sizeof(uint64_t));
  result = cudaMalloc((void**)&d_W,N*N*sizeof(uint64_t));
	assert(result == cudaSuccess);
  h_WInv = (uint64_t*)malloc(N*N*sizeof(uint64_t));
  result = cudaMalloc((void**)&d_WInv,N*N*sizeof(uint64_t));
	assert(result == cudaSuccess);

  // Computes W
  for(int j = 0; j < N; j++)
    for(int i = 0; i < N; i++)
        // h_W[i+j*N] = (( j == 0)? 1:(h_W[i-1+j*N]*pow(wN,i)%q));
        h_W[i+j*N] = powerMod(wN,j*i,P);

  for(int j = 0; j < N; j++)
    for(int i = 0; i < N; i++)
        h_WInv[i+j*N] = invMod(h_W[i+j*N],P);

  assert(h_W[1+N] == wN % P);
  assert(h_W[2+N] == powerMod(wN,2,P) % P);
  assert(h_W[2+2*N] == powerMod(wN,4,P) % P);

  std::cout << h_W[1+N] << " * " << h_WInv[1+N] << std::endl;
  std::cout << mulmod(h_W[1+N],h_WInv[1+N],P) << std::endl;
  assert(mulmod(h_W[1+N],h_WInv[1+N],P) == 1);
  assert(mulmod(h_W[2+N],h_WInv[2+N],P) == 1);
  assert(mulmod(h_W[2+2*N],h_WInv[2+2*N],P) == 1);

	// Generates random values
  for(int j = 0; j < NPOLYS;j++)
  	for(int i = 0; i < N/2; i++)
      h_a[i+j*NPOLYS] = i;
  		// h_a[i+j*NPOLYS] = rand() % q;

  // std::cout << "Input: " << std::endl;
  // for(int i = 0; i < N; i++)
  //   std::cout << h_a[i] << std::endl;

  // std::cout << "W: " << std::endl;

  // for(int i = 0; i < N; i++)
  //   std::cout << h_W[i] << std::endl;
  // for(int i = 0; i < N; i++)
  //   std::cout << h_W[i+1*N] << std::endl;

	// Copy to GPU
  result = cudaMemcpy(d_a,h_a , N*NPOLYS*sizeof(uint64_t), cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);

  result = cudaMemset((void*)d_b,0,N*NPOLYS*sizeof(uint64_t));

  result = cudaMemcpy(d_W,h_W , N*N*sizeof(uint64_t), cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);
  result = cudaMemcpy(d_WInv,h_WInv , N*N*sizeof(uint64_t), cudaMemcpyHostToDevice);
  assert(result == cudaSuccess);

	// Applies NTT
  // Foward
  NTT<<<gridDim,blockDim>>>(d_W,d_a,d_b,N,NPOLYS,P);
  assert(cudaGetLastError() == cudaSuccess);

  // result = cudaMemcpy(h_b,d_b ,  N*NPOLYS*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  // assert(result == cudaSuccess);
  // for(int i = 0; i < N; i++)
  //   std::cout << h_b[i] << std::endl;

  result = cudaMemset((void*)d_a,0,N*NPOLYS*sizeof(uint64_t));
  // Inverse
  NTT<<<gridDim,blockDim>>>(d_WInv,d_b,d_a,N,NPOLYS,P);
  assert(cudaGetLastError() == cudaSuccess);

	// Verify if the values were really shuffled
  result = cudaMemcpy(h_b,d_a,  N*NPOLYS*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	//
  // std::cout << "Output: " << std::endl;
  int count = 0;
  for(int i = 0; i < N; i++)
    if(h_b[i]/N != h_a[i])
    // std::cout << i << ") "<<h_b[i]/N << " != " << h_a[i] << std::endl;
      count++;
  std::cout << count << " errors." << std::endl;
	cudaFree(d_a);
	free(h_a);
	free(h_b);
  	std::cout << "Done." << std::endl;
}
