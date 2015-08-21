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
// #include "cuda_functios.h"

NTL_CLIENT

#define BILLION  1000000000L
#define MILLION  1000000L
#define DEGREE 128
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


// fast truncation of double-precision to integers
#define CUMP_D2I_TRUNC (double)(3ll << 51)
// computes r = a + b subop c unsigned using extended precision
#define VADDx(r, a, b, c, subop) \
    asm volatile("vadd.u32.u32.u32." subop " %0, %1, %2, %3;" :  \
            "=r"(r) : "r"(a) , "r"(b), "r"(c));

typedef struct {
  unsigned long long int lo;
  unsigned long long int hi;
} my_uint128;

__device__ my_uint128 add_uint128 (my_uint128 a, my_uint128 b)
{
  my_uint128 res;
  res.lo = a.lo + b.lo;
  res.hi = a.hi + b.hi + (res.lo < a.lo);
  return res;
} 

__device__ my_uint128 add_uint64_128 (uint64_t a, my_uint128 b)
{
  my_uint128 res = {a+b.lo,b.hi};
  return res;
} 

__device__ my_uint128 add_uint64_64 (uint64_t a, uint64_t b)
{

  my_uint128 res;
  res.lo = a + b;
  res.hi = (res.lo < a);

  return res;
} 


__device__ uint64_t s_rem (uint64_t a)
{
  // Special reduction for prime 2^64-2^32+1
  //
  // x3 * 2^96 + x2 * 2^64 + x1 * 2^32 + x0 \equiv
  // (x1+x2) * 2^32 + x0 - x3 - x2 mod p
  //
  // Here: x3 = 0, x2 = 0, x1 = (a >> 32), x0 = a-(x1 << 32)
  // const uint64_t p = 0xffffffff00000001;
  // uint64_t x3 = 0;
  // uint64_t x2 = 0;
  uint64_t x1 = (a >> 32);
  uint64_t x0 = (a & UINT32_MAX);

  // uint64_t res = ((x1+x2)<<32 + x0-x3-x2);
  uint64_t res = ((x1<<32) + x0);

  return res;
} 


__device__ uint64_t s_rem (my_uint128 a)
{
  // Special reduction for prime 2^64-2^32+1
  //
  // x3 * 2^96 + x2 * 2^64 + x1 * 2^32 + x0 \equiv
  // (x1+x2) * 2^32 + x0 - x3 - x2 mod p
  //
  // Here: x3 = 0, x2 = a.hi, x1 = (a.lo >> 32), x0 = a.lo-(x1 << 32)
  // const uint64_t p = 0xffffffff00000001;
  // uint64_t x3 = 0;
  uint64_t x2 = a.hi;
  uint64_t x1 = (a.lo >> 32);
  uint64_t x0 = (a.lo & UINT32_MAX);

  // uint64_t res = ((x1+x2)<<32 + x0-x3-x2);
  uint64_t res = (((x1+x2)<<32) + x0-x2);

  return res;
} 


__global__ void NTT64(uint64_t *W,uint64_t *a, uint64_t *a_hat, int N,int NPolis,uint64_t P){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
   const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int residueid = tid / (N);
  const int roffset = residueid*N;
  const int cid = tid & (N-1); // Coefficient id
  // const uint64_t p = 0xffffffff00000001;

  uint64_t *w;
  // if(type == FORWARD)
    w = W;
  // else
    // w = WInv;

  if(tid < N*NPolis){
    uint64_t value = 0;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      value = s_rem(add_uint64_64(value,mulmod(W64,a64,P)));

    }
    // if(type == FORWARD)
      a_hat[cid+roffset] = (value%P);
    // my_uint128 value128 = {value,0};
      // a_hat[cid+roffset] = s_rem(value128);
    // }
    // else
      // a_hat[cid+roffset] = (value)/N;

  }

}



__global__ void DOUBLENTT64(uint64_t *W,uint64_t *a, uint64_t *a_hat,uint64_t *b, uint64_t *b_hat, int N,int NPolis,uint64_t P){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int residueid = tid / N;
  const int roffset = residueid*N;
  const int cid = tid & (N-1); // Coefficient id
  uint64_t *w;
  // if(type == FORWARD)
    w = W;
  // else
    // w = WInv;

  // const uint64_t p = 0xffffffff00000001;
  if(tid < N*NPolis){
    uint64_t Avalue = 0;
    uint64_t Bvalue = 0;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){

      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      uint64_t b64 = b[i + roffset];
      Avalue = s_rem(add_uint64_64(Avalue,mulmod(W64,a64,P)));      
      Bvalue = s_rem(add_uint64_64(Bvalue,mulmod(W64,b64,P)));
    }
    // if(type == FORWARD){
      a_hat[cid+ roffset] = Avalue ;
      b_hat[cid+ roffset] = Bvalue ;
    // }else{
      // a_hat[cid+ roffset] = (Avalue )/N;
      // b_hat[cid+ roffset] = (Bvalue )/N;

      // a_hat[cid+ roffset] = (Avalue %P )/N;
      // b_hat[cid+ roffset] = (Bvalue %P )/N;
    // }
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
//     // actually uint64_t would do, but the casting is annoying
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
  const uint64_t P = 18446744069414584321;
  // const uint64_t P = 9223372036854829057;//63 bits
  // const uint64_t P = 4294955009;//63 bits
  // assert((P-1)%(N) == 0);
  // const uint64_t k = (P-1)/N;
  // const uint64_t wN = powerMod(3,k,P);
  ZZ PZZ = conv<ZZ>("18446744069414584321");
  uint64_t k = conv<uint64_t>(PZZ-1)/N;
  ZZ wNZZ = NTL::PowerMod(ZZ(7),k,PZZ);
  // assert((P-1)%(N) == 0);
  // const uint64_t k = (P-1)/N;
  const uint64_t wN = conv<uint64_t>(wNZZ);
  // wN = 17870292113338400769;

  // const uint64_t wN = 549755813888;// Hard coded
  // const uint64_t wN = 6209464568650184525;// Hard coded
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
        h_W[i+j*N] = conv<uint64_t>(NTL::PowerMod(wNZZ,j*i,PZZ));

  for(int j = 0; j < N; j++)
    for(int i = 0; i < N; i++)
        h_WInv[i+j*N] = conv<uint64_t>(NTL::InvMod(conv<ZZ>(h_W[i+j*N]),PZZ ));

  assert(h_W[1+N] == wN);
  assert(h_W[2+N] == NTL::PowerMod(wNZZ,2,PZZ));
  assert(h_W[2+2*N] == NTL::PowerMod(wNZZ,4,PZZ));

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
  NTT64<<<gridDim,blockDim>>>(d_W,d_a,d_b,N,NPOLYS,P);
  assert(cudaGetLastError() == cudaSuccess);

  // result = cudaMemcpy(h_b,d_b ,  N*NPOLYS*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  // assert(result == cudaSuccess);
  // for(int i = 0; i < N; i++)
  //   std::cout << h_b[i] << std::endl;

  result = cudaMemset((void*)d_a,0,N*NPOLYS*sizeof(uint64_t));
  // Inverse
  NTT64<<<gridDim,blockDim>>>(d_WInv,d_b,d_a,N,NPOLYS,P);
  assert(cudaGetLastError() == cudaSuccess);

	// Verify if the values were really shuffled
  result = cudaMemcpy(h_b,d_a,  N*NPOLYS*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	//
  // std::cout << "Output: " << std::endl;
  // int count = 0;
  for(int i = 0; i < N; i++)
    if(h_b[i]/N != h_a[i])
    std::cout << i << ") "<<h_b[i]/N << " != " << h_a[i] << std::endl;
      // count++;
  // std::cout << count << " errors." << std::endl;
	cudaFree(d_a);
	free(h_a);
	free(h_b);
  	std::cout << "Done." << std::endl;
}
