#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <NTL/ZZ.h>

NTL_CLIENT 

#define BILLION  1000000000L
#define MILLION  1000000L
// #define NITERATIONS 1
#define NITERATIONS 100
#define FIRSTITERATION 1024
#define LASTITERATION 1024
#define ADDBLOCKXDIM 128

enum ntt_mode_t {INVERSE,FORWARD};


double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

typedef uint64_t inteiro;

///////////////////////////////////////
/// ADD
__global__ void polynomialAdd(const inteiro *a,const inteiro *b,inteiro *c,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  inteiro a_value;
  inteiro b_value;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = a[tid];
      b_value = b[tid];

      a_value += b_value;

      c[tid] = a_value;
  }
}

///////////////////////////////////////
/// Mul

__global__ void polynomialNTTMul(const inteiro *a,const inteiro *b,inteiro *c,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  inteiro a_value;
  inteiro b_value;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = a[tid];
      b_value = b[tid];

      a_value *= b_value;

      c[tid] = a_value;
  }
}

__global__ void NTT(inteiro *W,inteiro *a, inteiro *a_hat, int N,int NPolis){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int cid = tid % N; // Coefficient id

  // const inteiro p = 0xffffffff00000001;
  if(tid < N*NPolis){

    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      inteiro value = W[cid + i*N]*a[cid];
      // sumReduce(W[cid + i*N]*a[cid],a_hat,i,1,N,NPolis);
      atomicAdd((unsigned long long int*)(&(a_hat[i])),(unsigned long long int)(value));
      __syncthreads();
    }
  }

}


__device__ uint64_t mulmod(uint64_t a, uint64_t b, uint64_t m) {
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
    __syncthreads();
    }
    return res;
}


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

__device__ my_uint128 mul_uint64_128 (uint64_t a, uint64_t b)
{
  my_uint128 res = {a*b,__umul64hi(a,b)};
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
  uint64_t res = ((x1+x2)<<32 + x0-x2);

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

__global__ void DOUBLENTT2(inteiro *W,inteiro *WInv,inteiro *a, inteiro *a_hat,inteiro *b, inteiro *b_hat, int N,int NPolis,inteiro P,const int type){
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
  if(type == FORWARD)
    w = W;
  else
    w = WInv;

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
      // Avalue = (Avalue + W64*a64);      
      // Bvalue = (Bvalue + W64*b64);
    }
    if(type == FORWARD){
      a_hat[cid+ roffset] = Avalue %P;
      b_hat[cid+ roffset] = Bvalue %P;
    }else{
      a_hat[cid+ roffset] = (Avalue %P)/N;
      b_hat[cid+ roffset] = (Bvalue %P)/N;
    }
  }
}

__global__ void NTT2(inteiro *W,inteiro *WInv,inteiro *a, inteiro *a_hat, int N,int NPolis,inteiro P,const int type){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

   const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int residueid = tid / (N);
  const int roffset = residueid*N;
  const int cid = tid & (N-1); // Coefficient id
  // const uint64_t p = 0xffffffff00000001;

  uint64_t *w;
  if(type == FORWARD)
    w = W;
  else
    w = WInv;

  if(tid < N*NPolis){
    uint64_t value = 0;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      value = s_rem(add_uint64_64(value,mulmod(W64,a64,P)));
    }
    if(type == FORWARD)
      a_hat[cid+roffset] = (value%P);
    else
      a_hat[cid+roffset] = (value%P)/N;

  }

}


__global__ void DOUBLENTT2C(inteiro *W,inteiro *WInv,inteiro *a, inteiro *a_hat,inteiro *b, inteiro *b_hat, int N,int NPolis,inteiro P,const int type){
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
  if(type == FORWARD)
    w = W;
  else
    w = WInv;

  // const uint64_t p = 0xffffffff00000001;
  if(tid < N*NPolis){
    uint64_t Avalue = 0;
    uint64_t Bvalue = 0;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){

      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      uint64_t b64 = b[i + roffset];
      // Avalue = (Avalue + W64*a64)%P;      
      // Bvalue = (Bvalue + W64*b64)%P;
      Avalue = (Avalue + W64*a64);      
      Bvalue = (Bvalue + W64*b64);
    }
    if(type == FORWARD){
      // a_hat[cid+ roffset] = Avalue % P;
      // b_hat[cid+ roffset] = Bvalue % P;
      a_hat[cid+ roffset] = Avalue ;
      b_hat[cid+ roffset] = Bvalue ;
    }else{
      // a_hat[cid+ roffset] = (Avalue % P)/N;
      // b_hat[cid+ roffset] = (Bvalue % P)/N;
      a_hat[cid+ roffset] = (Avalue );
      b_hat[cid+ roffset] = (Bvalue );
    }
  }
}

__global__ void NTT2C(inteiro *W,inteiro *WInv,inteiro *a, inteiro *a_hat, int N,int NPolis,inteiro P,const int type){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

 
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int residueid = tid / (N);
  const int roffset = residueid*N;
  const int cid = tid & (N-1); // Coefficient id
  // const double invk = (double)(1<<30) / P;
  uint64_t *w;
  if(type == FORWARD)
    w = W;
  else
    w = WInv;

  // const inteiro p = 0xffffffff00000001;
  if(tid < N*NPolis){
    uint64_t value = 0;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      // value = (value + W64*a64)%P;
      value = (value + W64*a64);
      // value = value + mul_m(W64,a64,P,invk);
    }
    if(type == FORWARD)
      // a_hat[cid+roffset] = value % P;
      a_hat[cid+roffset] = value ;
    else
      // a_hat[cid+roffset] = (value % P)/N;
      a_hat[cid+roffset] = (value );

  }

}



__global__ void NTT3(inteiro *W,inteiro *a, inteiro *a_hat, int N,int NPolis){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int cid = tid % N; // Coefficient id
  // __shared__ int local_W[ADDBLOCKXDIM];
  __shared__ inteiro local_a[ADDBLOCKXDIM];
  int loaded_index = 0;
  int j = 0;
  // const inteiro p = 0xffffffff00000001;
  if(tid < N*NPolis){
    inteiro value;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      if(i == loaded_index){
        // __syncthreads();
        if(threadIdx.x+i < N){

          // Load the next ADDBLOCKXDIM elements to shared memory
          // local_W[threadIdx.x + i] = W[(threadIdx.x + i) + cid*N];
          local_a[threadIdx.x] = a[threadIdx.x + i];
          // __syncthreads();
        }
        loaded_index += ADDBLOCKXDIM;
        j = 0;
      }
      value += W[i + cid*N]*local_a[j];
      j++;
    }
    a_hat[cid] = value;
  }

}


__global__ void DOUBLENTT4(inteiro *W,inteiro *WInv,inteiro *a, inteiro *a_hat,inteiro *b, inteiro *b_hat, int N,int NPolis,inteiro P,const int type){
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
  if(type == FORWARD)
    w = W;
  else
    w = WInv;

  // const uint64_t p = 0xffffffff00000001;
  if(tid < N*NPolis){
    my_uint128 Avalue = {0,0};
    my_uint128 Bvalue = {0,0};
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){

      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      uint64_t b64 = b[i + roffset];

      Avalue = (add_uint128(Avalue, mul_uint64_128(W64,a64)));      
      Bvalue = (add_uint128(Bvalue, mul_uint64_128(W64,b64)));
      // Avalue = (Avalue + W64*a64);      
      // Bvalue = (Bvalue + W64*b64);
    }
    if(type == FORWARD){
      a_hat[cid+ roffset] = s_rem(Avalue);
      b_hat[cid+ roffset] = s_rem(Bvalue);
      // a_hat[cid+ roffset] = Avalue ;
      // b_hat[cid+ roffset] = Bvalue ;
    }else{
      a_hat[cid+ roffset] = s_rem(Avalue)/N;
      b_hat[cid+ roffset] = s_rem(Bvalue)/N;
      // a_hat[cid+ roffset] = (Avalue );
      // b_hat[cid+ roffset] = (Bvalue );
    }
  }
}


__global__ void NTT4(inteiro *W,inteiro *WInv,inteiro *a, inteiro *a_hat, int N,int NPolis,inteiro P,const int type){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

   const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int residueid = tid / (N);
  const int roffset = residueid*N;
  const int cid = tid & (N-1); // Coefficient id
  // const uint64_t p = 0xffffffff00000001;

  uint64_t *w;
  if(type == FORWARD)
    w = W;
  else
    w = WInv;

  if(tid < N*NPolis){
    my_uint128 value = {0,0};
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      __syncthreads();
      value = (add_uint128(value, mul_uint64_128(W64,a64)));
    }
    if(type == FORWARD)
      a_hat[cid+roffset] = s_rem(value);
    else
      a_hat[cid+roffset] = s_rem(value)/N;

  }

}

__global__ void DOUBLENTT5(inteiro *W,inteiro *WInv,inteiro *a, inteiro *a_hat,inteiro *b, inteiro *b_hat, int N,int NPolis,inteiro P,const int type){
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
  if(type == FORWARD)
    w = W;
  else
    w = WInv;

  // const uint64_t p = 0xffffffff00000001;
  if(tid < N*NPolis){
    my_uint128 Avalue = {0,0};
    my_uint128 Bvalue = {0,0};
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){

      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      uint64_t b64 = b[i + roffset];

      Avalue = (add_uint128(Avalue, mul_uint64_128(W64,a64)));      
      Bvalue = (add_uint128(Bvalue, mul_uint64_128(W64,b64)));
      // Avalue = (Avalue + W64*a64);      
      // Bvalue = (Bvalue + W64*b64);
    }
    if(type == FORWARD){
      a_hat[cid+ roffset] = s_rem(Avalue);
      b_hat[cid+ roffset] = s_rem(Bvalue);
      // a_hat[cid+ roffset] = Avalue ;
      // b_hat[cid+ roffset] = Bvalue ;
    }else{
      a_hat[cid+ roffset] = s_rem(Avalue);
      b_hat[cid+ roffset] = s_rem(Bvalue);
      // a_hat[cid+ roffset] = (Avalue );
      // b_hat[cid+ roffset] = (Bvalue );
    }
  }
}


__global__ void NTT5(inteiro *W,inteiro *WInv,inteiro *a, inteiro *a_hat, int N,int NPolis,inteiro P,const int type){
  // This algorithm supposes that N is power of 2, divisible by 32
  // Input:
  // w: Matrix of wNs
  // a: residues
  // a_hat: output
  // N: # of coefficients of each polynomial
  // NPolis: # of residues

   const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int residueid = tid / (N);
  const int roffset = residueid*N;
  const int cid = tid & (N-1); // Coefficient id
  // const uint64_t p = 0xffffffff00000001;

  uint64_t *w;
  if(type == FORWARD)
    w = W;
  else
    w = WInv;

  if(tid < N*NPolis){
    my_uint128 value = {0,0};
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      uint64_t W64 = w[i + cid*N];
      uint64_t a64 = a[i + roffset];
      value = (add_uint128(value, mul_uint64_128(W64,a64)));
    }
    if(type == FORWARD)
      a_hat[cid+roffset] = s_rem(value);
    else
      a_hat[cid+roffset] = s_rem(value);

  }

}

inteiro powerMod(inteiro x,long h,inteiro p){
  unsigned long t;
  if(h == 0)
    return 1;
  else if(h == 1)
    return x % p;
  else
    t = log2((double)(h))+1;
  ZZ r = ZZ(x);
  ZZ X = ZZ(x);
  ZZ P = ZZ(p);

  for(int i = t-1; i >= 0; i--){
    r = r*r;
    r %= P;
    if((h >> i) & 1 == 1)//i-th bit
      r *= X % P;
    
  }
  return conv<inteiro>(r);
}

inteiro invMod(inteiro x,inteiro p){
  // We suppose that p is a prime
  return powerMod(x,p-2,p);
}
int main(){
  // std::cout << "Add: "<< std::endl << std::endl;
  // for(int degree = 128; degree <= 2048; degree *= 2){
  //  // ADD

  //  inteiro* h_a;
  //  inteiro* h_b;
  //  inteiro* d_a;
  //  inteiro* d_b;
  //  inteiro* d_c;
  //  int N = degree+1;
  //  struct timespec start, stop;

  //  h_a = (inteiro*)malloc(N*sizeof(inteiro));
  //  h_b = (inteiro*)malloc(N*sizeof(inteiro));

  //  for(int i = 0; i < N;i++){
  //    h_a[i] = rand();
  //    h_b[i] = rand();
  //  }
  //  cudaError_t result;
  //  clock_gettime( CLOCK_REALTIME, &start);
  //  for(int i = 0; i < NITERATIONS;i++){

  //    result = cudaMalloc((void**)&d_a,N*sizeof(inteiro));
  //    assert(result == cudaSuccess);
  //    result = cudaMalloc((void**)&d_b,N*sizeof(inteiro));
  //    assert(result == cudaSuccess);
  //    result = cudaMalloc((void**)&d_c,N*sizeof(inteiro));
  //    assert(result == cudaSuccess);

  //    result = cudaMemcpy(d_a,h_a , N*sizeof(inteiro), cudaMemcpyHostToDevice);
  //    assert(result == cudaSuccess);
  //    result = cudaMemcpy(d_b,h_b , N*sizeof(inteiro), cudaMemcpyHostToDevice);
  //    assert(result == cudaSuccess);
  //    result = cudaDeviceSynchronize();
  //    assert(result == cudaSuccess);
  //  }
  //  clock_gettime( CLOCK_REALTIME, &stop);
  //  std::cout << degree <<") Alloc and copy time 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl;

  //  dim3 blockDim(ADDBLOCKXDIM);
  //  dim3 gridDim(N/ADDBLOCKXDIM);

  //  clock_gettime( CLOCK_REALTIME, &start);
  //  for(int i = 0; i < NITERATIONS;i++){
  //    polynomialAdd<<<gridDim,blockDim>>>(d_a,d_b,d_c,N);
  //    result = cudaDeviceSynchronize();
  //    assert(result == cudaSuccess);
  //  }
  //  clock_gettime( CLOCK_REALTIME, &stop);
  //  std::cout << degree <<") Add kernel 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl;

  // }


  std::cout << "Mul: "<< std::endl << std::endl;
  for(int degree = FIRSTITERATION; degree <= LASTITERATION; degree *= 2){
    // MUL
    std::cout << "###################################" << std::endl;

    // const inteiro P = 4294955009;
    const inteiro P = 18446744069414584321;
    assert((P-1)%(degree) == 0);
    const inteiro k = (P-1)/(degree);
    const inteiro wN = powerMod(3,k,P);
    inteiro* h_W;
    inteiro* h_WInv;
    inteiro* d_W;
    inteiro* d_WInv;

    inteiro* h_a;
    inteiro* h_b;
    inteiro* d_a;
    inteiro* d_b;
    inteiro* d_c;
    inteiro* d_A;
    inteiro* d_B;
    inteiro* d_C;
    int N = degree+1;
    struct timespec start, stop;

    h_a = (inteiro*)malloc(N*sizeof(inteiro));
    h_b = (inteiro*)malloc(N*sizeof(inteiro));

    for(int i = 0; i < N;i++){
      if(i < N/2){
        h_a[i] = rand();
        h_b[i] = rand();
      }else{
        h_a[i] = 0;
        h_b[i] = 0;
      }
    }
    cudaError_t result;

    h_W = (inteiro*)malloc(N*N*sizeof(inteiro));
    result = cudaMalloc((void**)&d_W,N*N*sizeof(inteiro));
    assert(result == cudaSuccess);
    h_WInv = (inteiro*)malloc(N*N*sizeof(inteiro));
    result = cudaMalloc((void**)&d_WInv,N*N*sizeof(inteiro));
    assert(result == cudaSuccess);

    // Computes W
    for(int j = 0; j < N; j++)
      for(int i = 0; i < N; i++)
        // h_W[i+j*N] = (( j == 0)? 1:(h_W[i-1+j*N]*pow(wN,i)%q));
        if(i == 0)
          h_W[i+j*N] = 1; 
        else
          h_W[i+j*N] = h_W[(i-1)+j*N]*powerMod(wN,j,P) % P;

    inteiro NInv = invMod(N,P);
    for(int j = 0; j < N; j++)
      for(int i = 0; i < N; i++)
        h_WInv[i+j*N] = invMod(h_W[i+j*N],P)*NInv % P;

    clock_gettime( CLOCK_REALTIME, &start);
    for(int i = 0; i < NITERATIONS;i++){

      result = cudaMalloc((void**)&d_a,N*sizeof(inteiro));
      assert(result == cudaSuccess);
      result = cudaMalloc((void**)&d_b,N*sizeof(inteiro));
      assert(result == cudaSuccess);
      result = cudaMalloc((void**)&d_c,N*sizeof(inteiro));
      assert(result == cudaSuccess);
      result = cudaMalloc((void**)&d_A,N*sizeof(inteiro));
      assert(result == cudaSuccess);
      result = cudaMalloc((void**)&d_B,N*sizeof(inteiro));
      assert(result == cudaSuccess);
      result = cudaMalloc((void**)&d_C,N*sizeof(inteiro));
      assert(result == cudaSuccess);

      result = cudaMemcpy(d_a,h_a , N*sizeof(inteiro), cudaMemcpyHostToDevice);
      assert(result == cudaSuccess);
      result = cudaMemcpy(d_b,h_b , N*sizeof(inteiro), cudaMemcpyHostToDevice);
      assert(result == cudaSuccess);

      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
    }
    clock_gettime( CLOCK_REALTIME, &stop);
    std::cout << degree <<") Alloc and copy time 32 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl;

    dim3 blockDim(ADDBLOCKXDIM);
    dim3 gridDim(N/ADDBLOCKXDIM);

    // std:: cout << "4 kernels" << std::endl;
    // clock_gettime( CLOCK_REALTIME, &start);
    // for(int i = 0; i < NITERATIONS;i++){
    //   NTT<<<gridDim,blockDim>>>(d_W,d_a,d_A,N,1); // Forward
    //   NTT<<<gridDim,blockDim>>>(d_W,d_b,d_B,N,1); // Forward
    //   polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
    //   NTT<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse


    //   result = cudaDeviceSynchronize();
    //   assert(result == cudaSuccess);
    // }
    // clock_gettime( CLOCK_REALTIME, &stop);
    // std::cout << degree <<") NTT1 Mul kernel 32 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

      std:: cout << "4 kernels" << std::endl;
      clock_gettime( CLOCK_REALTIME, &start);
      for(int i = 0; i < NITERATIONS;i++){
        NTT2<<<gridDim,blockDim>>>(d_W,d_WInv,d_a,d_A,N,1,P,FORWARD); // Forward
        NTT2<<<gridDim,blockDim>>>(d_W,d_WInv,d_b,d_B,N,1,P,FORWARD); // Forward
        polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
        NTT2<<<gridDim,blockDim>>>(d_W,d_WInv,d_c,d_C,N,1,P,INVERSE);// Inverse


        result = cudaDeviceSynchronize();
        assert(result == cudaSuccess);
      }
      clock_gettime( CLOCK_REALTIME, &stop);
      std::cout << degree <<") NTT2 Mul kernel 32 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

    std:: cout << "4 kernels" << std::endl;
      clock_gettime( CLOCK_REALTIME, &start);
      for(int i = 0; i < NITERATIONS;i++){
        DOUBLENTT2<<<gridDim,blockDim>>>(d_W,d_WInv,d_a,d_A,d_b,d_B,N,1,P,FORWARD); // Forward
        polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
        NTT2<<<gridDim,blockDim>>>(d_W,d_WInv,d_c,d_C,N,1,P,INVERSE);// Inverse


        result = cudaDeviceSynchronize();
        assert(result == cudaSuccess);
      }
      clock_gettime( CLOCK_REALTIME, &stop);
      std::cout << degree <<") DOUBLENTT2 Mul kernel 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;


      std:: cout << "4 kernels" << std::endl;
      clock_gettime( CLOCK_REALTIME, &start);
      for(int i = 0; i < NITERATIONS;i++){
        NTT2C<<<gridDim,blockDim>>>(d_W,d_WInv,d_a,d_A,N,1,P,FORWARD); // Forward
        NTT2C<<<gridDim,blockDim>>>(d_W,d_WInv,d_b,d_B,N,1,P,FORWARD); // Forward
        polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
        NTT2C<<<gridDim,blockDim>>>(d_W,d_WInv,d_c,d_C,N,1,P,INVERSE);// Inverse


        result = cudaDeviceSynchronize();
        assert(result == cudaSuccess);
      }
      clock_gettime( CLOCK_REALTIME, &stop);
      std::cout << degree <<") NTT2C Mul kernel 32 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;


    std:: cout << "4 kernels" << std::endl;
      clock_gettime( CLOCK_REALTIME, &start);
      for(int i = 0; i < NITERATIONS;i++){
        DOUBLENTT2C<<<gridDim,blockDim>>>(d_W,d_WInv,d_a,d_A,d_b,d_B,N,1,P,FORWARD); // Forward
        polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
        NTT2C<<<gridDim,blockDim>>>(d_W,d_WInv,d_c,d_C,N,1,P,INVERSE);// Inverse


        result = cudaDeviceSynchronize();
        assert(result == cudaSuccess);
      }
      clock_gettime( CLOCK_REALTIME, &stop);
      std::cout << degree <<") DOUBLENTT2C Mul kernel 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

      // std:: cout << "4 kernels" << std::endl;
      // clock_gettime( CLOCK_REALTIME, &start);
      // for(int i = 0; i < NITERATIONS;i++){
      //   NTT3<<<gridDim,blockDim>>>(d_W,d_a,d_A,N,1); // Forward
      //   NTT3<<<gridDim,blockDim>>>(d_W,d_b,d_B,N,1); // Forward
      //   polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
      //   NTT3<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse


      //   result = cudaDeviceSynchronize();
      //   assert(result == cudaSuccess);
      // }
      // clock_gettime( CLOCK_REALTIME, &stop);
      // std::cout << degree <<") NTT3 Mul kernel 32 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;
     
      std:: cout << "4 kernels" << std::endl;
      clock_gettime( CLOCK_REALTIME, &start);
      for(int i = 0; i < NITERATIONS;i++){
        NTT4<<<gridDim,blockDim>>>(d_W,d_WInv,d_a,d_A,N,1,P,FORWARD); // Forward
        NTT4<<<gridDim,blockDim>>>(d_W,d_WInv,d_a,d_A,N,1,P,FORWARD); // Forward
        polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
        NTT4<<<gridDim,blockDim>>>(d_W,d_WInv,d_c,d_C,N,1,P,INVERSE);// Inverse


        result = cudaDeviceSynchronize();
        assert(result == cudaSuccess);
      }
      clock_gettime( CLOCK_REALTIME, &stop);
      std::cout << degree <<") NTT4 Mul kernel 32 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

    std:: cout << "4 kernels" << std::endl;
      clock_gettime( CLOCK_REALTIME, &start);
      for(int i = 0; i < NITERATIONS;i++){
        DOUBLENTT4<<<gridDim,blockDim>>>(d_W,d_WInv,d_a,d_A,d_b,d_B,N,1,P,FORWARD); // Forward
        polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
        NTT4<<<gridDim,blockDim>>>(d_W,d_WInv,d_c,d_C,N,1,P,INVERSE);// Inverse


        result = cudaDeviceSynchronize();
        assert(result == cudaSuccess);
      }
      clock_gettime( CLOCK_REALTIME, &stop);
      std::cout << degree <<") DOUBLENTT4 Mul kernel 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

    // std:: cout << "4 kernels" << std::endl;
    //   clock_gettime( CLOCK_REALTIME, &start);
    //   for(int i = 0; i < NITERATIONS;i++){
    //     DOUBLENTT5<<<gridDim,blockDim>>>(d_W,d_WInv,d_a,d_A,d_b,d_B,N,1,P,FORWARD); // Forward
    //     polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
    //     NTT5<<<gridDim,blockDim>>>(d_W,d_WInv,d_c,d_C,N,1,P,INVERSE);// Inverse


    //     result = cudaDeviceSynchronize();
    //     assert(result == cudaSuccess);
    //   }
    //   clock_gettime( CLOCK_REALTIME, &stop);
    //   std::cout << degree <<") DOUBLENTT5 Mul kernel 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;



  }
}