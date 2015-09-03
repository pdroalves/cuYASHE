#include "cuda_functions.h"
#include "common.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>

cuyasheint_t CUDAFunctions::wN = 0;
cuyasheint_t *CUDAFunctions::d_W = NULL;
cuyasheint_t *CUDAFunctions::d_WInv = NULL;
int CUDAFunctions::N = 0;

///////////////////////////////////////
/// Memory operations
__global__ void realignCRTResidues(int oldSpacing,int newSpacing, cuyasheint_t *array,cuyasheint_t *new_array,int residuesSize,int residuesQty){
  //
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  const int residueId = tid / residuesSize;
  const int new_array_offset = (tid % residuesSize) + residueId*newSpacing;
  const int old_array_offset = (tid % residuesSize) + residueId*oldSpacing;

  if(new_array_offset < newSpacing*residuesQty && old_array_offset < oldSpacing*residuesQty )
    new_array[new_array_offset] = array[old_array_offset];

}


__host__ cuyasheint_t* CUDAFunctions::callRealignCRTResidues(cudaStream_t stream,int oldSpacing,int newSpacing, cuyasheint_t *array,int residuesSize,int residuesQty){
  if(oldSpacing == newSpacing)
    return NULL;
  const int size = residuesSize*residuesQty;
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  cuyasheint_t *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,newSpacing*residuesQty*sizeof(cuyasheint_t));
  #ifdef VERBOSE
  std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< newSpacing*residuesQty*sizeof(cuyasheint_t) << " bytes" <<std::endl;
  #endif
  assert(result == cudaSuccess);

  realignCRTResidues <<< gridDim,blockDim,1,stream >>> (oldSpacing,newSpacing,array,d_new_array,residuesSize,residuesQty);
  #ifdef VERBOSE
  std::cout << gridDim.x << " " << blockDim.x << std::endl;
  std::cout << "polynomialAdd kernel:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
  #endif
  assert(cudaGetLastError() == cudaSuccess);

  return d_new_array;
}

///////////////////////////////////////

///////////////////////////////////////
/// ADD
__global__ void polynomialAddSub(const int OP,const cuyasheint_t *a,const cuyasheint_t *b,cuyasheint_t *c,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  cuyasheint_t a_value;
  cuyasheint_t b_value;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = a[tid];
      b_value = b[tid];

      if(OP == ADD)
        a_value += b_value;
      else
        a_value -= b_value;

      c[tid] = a_value;
  }
}

__host__ cuyasheint_t* CUDAFunctions::callPolynomialAddSub(cudaStream_t stream,cuyasheint_t *a,cuyasheint_t *b,int size,int OP){
  // This method expects that both arrays are aligned
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  cuyasheint_t *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,size*sizeof(cuyasheint_t));
  #ifdef VERBOSE
  std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< size*sizeof(cuyasheint_t) << " bytes" <<std::endl;
  #endif
  assert(result == cudaSuccess);

  // polynomialAdd <<< gridDim,blockDim, 0, stream >>> (a,b,d_new_array,size);
  polynomialAddSub <<< gridDim,blockDim >>> (OP,a,b,d_new_array,size);
  #ifdef VERBOSE
  std::cout << gridDim.x << " " << blockDim.x << std::endl;
  std::cout << "polynomialAdd kernel:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
  #endif

  return d_new_array;
}
///////////////////////////////////////

///////////////////////////////////////
/// MUL

__global__ void polynomialPlainMul(const cuyasheint_t *a,const cuyasheint_t *b,cuyasheint_t *c,const int N,const int NPolis){
  // Each block computes one coefficient of c
  // We need 2*N blocks for each residue!
  // 2D-blocks

  // const int tidX = threadIdx.x + blockDim.x*blockIdx.x;
  // const int tidY = threadIdx.y + blockDim.y*blockIdx.y;

  // blockDim.x == blockDim.y
  // We suppose that TILEDIM divides 2*N
  const int TILEDIM = blockDim.x;
  const int coefficient = blockIdx.x % (2*N);
  const int residueID = blockIdx.x / (2*N);
  const int residueOffset = residueID*(2*N);
  __shared__ cuyasheint_t value;
  value = 0;

  // blockDim.x == blockDim.y
  // if(tidX < N && tidY < N){
    for(int tileY = 0;tileY < N/TILEDIM; tileY++)
      for(int tileX = 0;tileX < N/TILEDIM; tileX++){
        //      (           coefficient    ) + residue
        int i = (threadIdx.x + tileX*TILEDIM);
        int j = (threadIdx.y + tileY*TILEDIM);

        if(i + j == coefficient)
          atomicAdd((unsigned long long int *)(&value),(unsigned long long int)(a[i+residueOffset]*b[j+residueOffset]));
      }
    __syncthreads();

    if(threadIdx.x == 0 && threadIdx.y == 0)
      c[coefficient+residueOffset] = value;
      // There are 2N threads in Y axis computing this coefficient
      // atomicAdd((unsigned cuyasheint_t cuyasheint_t int*)(&(c[coefficient+residueOffset])),(unsigned cuyasheint_t cuyasheint_t int)(value));

  // }
}


__host__ __device__ inline cuyasheint_t s_rem (uint64_t a)
{
  uint64_t res = (a>>31) + (a&0x7FFFFFFF);
  // This implies in an annoying divergence
  if(res > 0x7FFFFFFF)
    res = (cuyasheint_t)((res>>31) + (res&0x7FFFFFFF));
  #ifdef __CUDA_ARCH__
  // Only for device code
  __syncthreads();
  #endif
  return (cuyasheint_t)res;
}

__host__ __device__ void butterfly(uint64_t *v){
	// Butterfly
	uint64_t v0 = v[0];
	v[0] = v0 + v[1];
	v[1] = v0 - v[1];
}

__host__ __device__ int expand(int idxL, int N1, int N2){
	return (idxL/N1)*N1*N2 + (idxL%N1);
}

__device__ __host__ void NTTIteration(cuyasheint_t *W,cuyasheint_t *WInv,const int j,const int N,const int R,const int Ns, const cuyasheint_t* data0, cuyasheint_t *data1, const int type){
	uint64_t v[2];
	int idxS = j;
	// int wIndex;
	cuyasheint_t *w;
	if(type == FORWARD){
		w = W;
	}else{
		w = WInv;
	}

	for(int r=0; r<R; r++){
		v[r] = ((uint64_t)data0[idxS+r*N/R])*w[j];
	}
	butterfly(v);
	int idxD = expand(j,Ns,R);
	for(int r=0; r<R;r++)
    if(type == FORWARD)
  		data1[idxD+r*Ns] = s_rem(v[r]);
    else
      data1[idxD+r*Ns] = s_rem(v[r])/2;

}

__global__ void NTT(cuyasheint_t *d_W,cuyasheint_t *d_WInv,const int N, const int R, const int Ns, cuyasheint_t* dataI, cuyasheint_t* dataO,const int type){

	  for(int i = 0; i < N/R; i += 1024){
    // " Threads virtuais "
    const int j = blockIdx.y*N + blockIdx.x*blockDim.x + threadIdx.x;
    if( j < N)
      NTTIteration(d_W,d_WInv,j, N, R, Ns, dataI, dataO,type);
  }
}

__global__ void polynomialNTTMul(cuyasheint_t *a,const cuyasheint_t *b,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  // const cuyasheint_t P = 0xffffffff00000001;
  uint64_t a_value;
  uint64_t b_value;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = a[tid];
      b_value = b[tid];

      // In-place
      a[tid] = s_rem(a_value*b_value);
  }
}

typedef float2 Complex;

#define PI_F 3.141592654f

static __device__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

static __device__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__global__ void FFT32(Complex *a, cuyasheint_t *a_hat, const int N,const int NPolis,const int type){
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
  Complex wN;
  if(type == FORWARD){
    wN.x = __cosf(2*PI_F/N);
    wN.y = __sinf(2*PI_F/N);
  }
  else{
    wN.x = __cosf(-2*PI_F/N);
    wN.y = __sinf(-2*PI_F/N);
  }
  Complex W64 = {1,0};

  if(tid < N*NPolis){
    // my_uint128 value = {0,0};
    Complex value = {0,0};
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      Complex a64 = a[i + roffset];
      value = ComplexAdd(value,ComplexMul(W64,a64));
      W64 = ComplexMul(W64,wN);
    }
    if(type == FORWARD)
      a_hat[cid+roffset] = (value.x);
    else
      a_hat[cid+roffset] = (value.x)/N;

  }

}

__global__ void DOUBLEFFT32( cuyasheint_t *a, Complex *a_hat,cuyasheint_t *b, Complex *b_hat, const int N,const int NPolis,const int type){
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
  Complex wN;
  if(type == FORWARD){
    wN.x = __cosf(2*PI_F/N);
    wN.y = __sinf(2*PI_F/N);
  }
  else{
    wN.x = __cosf(-2*PI_F/N);
    wN.y = __sinf(-2*PI_F/N);
  }
  Complex W64 = {1,0};

  if(tid < N*NPolis){
    Complex Avalue = {0,0};
    Complex Bvalue = {0,0};

    for(int i = 0; i < N; i++){
      // uint64_t a64 = a[((i + roffset+cid)&(N-1))];
      // uint64_t b64 = b[((i + roffset+cid)&(N-1))];
      Complex a64 = {(float)a[i + roffset],0};
      Complex b64 = {(float)b[i + roffset],0};
      Avalue = ComplexAdd(Avalue, ComplexMul(W64,a64));
      Bvalue = ComplexAdd(Bvalue, ComplexMul(W64,b64));
      W64 = ComplexMul(W64,wN);
    }
    if(type == FORWARD){
      a_hat[cid+ roffset] = (Avalue);
      b_hat[cid+ roffset] = (Bvalue);
    }else{
      a_hat[cid+ roffset] = {(Avalue.x)/N,0};
      b_hat[cid+ roffset] = {(Bvalue.x)/N,0};
    }
  }
}

__global__ void polynomialFFTMul(Complex *a,const Complex *b,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      // In-place
      a[tid] = ComplexMul(a[tid],b[tid]);
  }
}

__host__ cuyasheint_t* CUDAFunctions::callPolynomialMul(cudaStream_t stream,cuyasheint_t *a,cuyasheint_t *b,int N,int NPolis){
  // This method expects that both arrays are aligned

  // Input:
  // stream: cudaStream
  // a: first operand
  // b: second operand
  // N: number of coefficients for each operand
  // NPolis: number of residues
  // All representations should be concatenated aligned
  assert((N>0)&&((N & (N - 1)) == 0));//Check if N is power of 2
  assert(N == CUDAFunctions::N);
  cuyasheint_t *d_result;

  #ifdef PLAINMUL
    #ifdef VERBOSE
        std::cout << "Plain multiplication" << std::endl;
    #endif
    cudaError_t result = cudaMalloc((void**)&d_result,(N)*NPolis*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    result = cudaMemset((void*)d_result,0,(N)*NPolis*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

    dim3 blockDim(ADDBLOCKXDIM,ADDBLOCKXDIM);
    // int blocks = ((2*N*NPolis) % ADDBLOCKXDIM == 0? (2*N*NPolis)/ADDBLOCKXDIM : (2*N*NPolis)/ADDBLOCKXDIM+1);
    // dim3 gridDim(blocks,blocks);
    dim3 gridDim(N*NPolis,1);
    polynomialPlainMul<<<gridDim,blockDim,1,stream>>>(a,b,d_result,N,NPolis);
    assert(cudaGetLastError() == cudaSuccess);
  #elif defined(NTTMUL)
        // std::cout << "NTT multiplication" << std::endl;

    // Allocates memory for temporary arrays on device
    // Each polynomial's degree gets doubled
    //
    // d_result is used as auxiliar array
    const int size = N*NPolis;
    cuyasheint_t *d_a;
    cuyasheint_t *d_b;
    cudaError_t result = cudaMalloc((void**)&d_result,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_a,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_b,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

    const int RADIX = 2;
  	dim3 blockDim(std::max(1,N/(RADIX*1024)),NPolis);
    // dim3 gridDim((N/RADIX)/blockDim.x);
  	dim3 gridDim(1);

    // Forward
    for(int Ns=1; Ns<N; Ns*=RADIX){
      NTT<<<gridDim,blockDim,1,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,RADIX,Ns,a,d_a,FORWARD);
      assert(cudaGetLastError() == cudaSuccess);
    }
    for(int Ns=1; Ns<N; Ns*=RADIX){
      NTT<<<gridDim,blockDim,1,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,RADIX,Ns,b,d_b,FORWARD);
      assert(cudaGetLastError() == cudaSuccess);
    }
    // Multiply
    dim3 blockDimMul(ADDBLOCKXDIM);
    dim3 gridDimMul((size)/ADDBLOCKXDIM+1); // We expect that ADDBLOCKXDIM always divice size
    polynomialNTTMul<<<gridDimMul,blockDimMul,1,stream>>>(d_a,d_b,N*NPolis);
    assert(cudaGetLastError() == cudaSuccess);

    // Inverse
    for(int Ns=1; Ns<N; Ns*=RADIX){
      NTT<<<gridDim,blockDim,1,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,RADIX,Ns,d_a,d_result,INVERSE);
      assert(cudaGetLastError() == cudaSuccess);
    }
    cudaFree(d_a);
    cudaFree(d_b);
  #elif defined(FFTMUL)
    // Allocates memory for temporary arrays on device
    // Each polynomial's degree gets doubled
    const int size = N*NPolis;
    cudaError_t result = cudaMalloc((void**)&d_result,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

    Complex *d_A;
    Complex *d_B;
    result = cudaMalloc((void**)&d_A,size*sizeof(Complex));
    assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_B,size*sizeof(Complex));
    assert(result == cudaSuccess);

    dim3 blockDim(ADDBLOCKXDIM);
    dim3 gridDim((size)/ADDBLOCKXDIM+1); // We expect that ADDBLOCKXDIM always divice size

    assert(blockDim.x*gridDim.x >= N);
    // Forward
    DOUBLEFFT32<<<gridDim,blockDim,1,stream>>>(a,d_A,b,d_B,N,NPolis,FORWARD);
    assert(cudaGetLastError() == cudaSuccess);

    // Multiply
    polynomialFFTMul<<<gridDim,blockDim,1,stream>>>(d_A,d_B,N*NPolis);

    // Inverse
    FFT32<<<gridDim,blockDim,1,stream>>>(d_A,d_result,N,NPolis,INVERSE);
    assert(cudaGetLastError() == cudaSuccess);

    cudaFree(d_A);
    cudaFree(d_B);

  #endif

  return d_result;
}

__host__ void CUDAFunctions::init(int N){
    CUDAFunctions::N = N;

  #ifdef NTTMUL
    // #ifdef VERBOSE
    std::cout << "Will compute W." << std::endl;
    // #endif

    cuyasheint_t *h_W;
    cuyasheint_t *h_WInv;

    ZZ PZZ = conv<ZZ>("2147483647");
    cuyasheint_t k = conv<cuyasheint_t>(PZZ-1)/N;
    ZZ wNZZ = NTL::PowerMod(ZZ(7),k,PZZ);
    // assert((P-1)%(N) == 0);
    // const cuyasheint_t k = (P-1)/N;
    wN = conv<cuyasheint_t>(wNZZ);
    // wN = 17870292113338400769;

    cudaError_t result;
    h_W = (cuyasheint_t*)malloc(N*sizeof(cuyasheint_t));
    result = cudaMalloc((void**)&d_W,N*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    h_WInv = (cuyasheint_t*)malloc(N*sizeof(cuyasheint_t));
    result = cudaMalloc((void**)&d_WInv,N*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

    // Computes 1-th column from W
    for(int j = 0; j < N; j++)
        h_W[j] = conv<cuyasheint_t>(NTL::PowerMod(wNZZ,j,PZZ));

    // Computes 1-th column from WInv
    for(int j = 0; j < N; j++)
        h_WInv[j] = conv<cuyasheint_t>(NTL::InvMod(conv<ZZ>(h_W[j]),PZZ ));

    result = cudaMemcpy (d_W,h_W , N*sizeof(cuyasheint_t),cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);
    result = cudaMemcpy(d_WInv,h_WInv , N*sizeof(cuyasheint_t),cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    free(h_W);
    free(h_WInv);
    #endif
    }
