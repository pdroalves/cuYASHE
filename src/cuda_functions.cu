#include "cuda_functions.h"
#include "settings.h"
#include "polynomial.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>


#ifdef NTTMUL
cuyasheint_t CUDAFunctions::wN = 0;
cuyasheint_t *CUDAFunctions::d_W = NULL;
cuyasheint_t *CUDAFunctions::d_WInv = NULL;
#elif defined(CUFFTMUL)
cufftHandle CUDAFunctions::plan;
typedef double2 Complex;
#endif

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
  #ifdef VERBOSE
  std::cout << "Realigning..." << std::endl;
  #endif
  const int size = residuesSize*residuesQty;
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  cuyasheint_t *d_new_array;
  cudaError_t result = cudaMalloc((void**)&d_new_array,newSpacing*residuesQty*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);

  realignCRTResidues <<< gridDim,blockDim,1,stream >>> (oldSpacing,newSpacing,array,d_new_array,residuesSize,residuesQty);
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
  assert(result == cudaSuccess);

  polynomialAddSub <<< gridDim,blockDim,1,stream >>> (OP,a,b,d_new_array,size);
  #ifdef VERBOSE
  std::cout << gridDim.x << " " << blockDim.x << std::endl;
  std::cout << "polynomialAdd kernel:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
  #endif

  return d_new_array;
}

__host__ void CUDAFunctions::callPolynomialAddSubInPlace(cudaStream_t stream,cuyasheint_t *a,cuyasheint_t *b,int size,int OP){
  // This method expects that both arrays are aligned
  // Add and store in array a
  int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  dim3 gridDim(ADDGRIDXDIM);
  dim3 blockDim(ADDBLOCKXDIM);

  polynomialAddSub <<< gridDim,blockDim,1,stream >>> (OP,a,b,a,size);
  #ifdef VERBOSE
  std::cout << gridDim.x << " " << blockDim.x << std::endl;
  std::cout << "polynomialAdd kernel:" << cudaGetErrorString(cudaGetLastError()) << std::endl;
  #endif
}
///////////////////////////////////////

///////////////////////////////////////
/// MUL

#ifdef PLAINMUL
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
#elif defined(CUFFTMUL)

__global__ void copyIntegerToComplex(Complex *a,cuyasheint_t *b,int size){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size ){
      a[tid].x =  (b[tid]);
      // printf("%ld => %f\n\n",b[tid],a[tid].x);
      a[tid].y = 0;
  }else{
    a[tid].x = 0;
    a[tid].y = 0;
  }
}

__global__ void copyAndNormalizeComplexRealPartToInteger(cuyasheint_t *b,Complex *a,int size,double scale){
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  int value;
  double fvalue;
  // double frac;
  if(tid < size ){
      fvalue = a[tid].x * scale;
      value = rint(fvalue);
      // frac = fmodf(fvalue,1);
      // if( frac >= 0.5f)
      //   value += 1;

      // value =  (a[tid].x * scale);
      b[tid] = value;
      // printf("%f) %f  => %ld\n\n",a[tid].x, a[tid].x * scale,value);

      // b[tid] =  __ddouble2ull_rd(a[tid].x/scale);
  }
}
////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex multiplication
static __device__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}


// Complex pointwise multiplication
static __global__ void polynomialcuFFTMul(const Complex *a, const Complex *b,Complex *c,int size_c,int size)
{
    const int tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(tid < size  ){
        c[tid] = ComplexMul(a[tid], b[tid]);
    }else{
      c[tid].x = 0;
      c[tid].y = 0;
    }
}
#elif defined(NTTMUL)
__device__ __host__ uint64_t s_rem (uint64_t a)
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

__device__ __host__  uint64_t s_mul(uint64_t a,uint64_t b){
  // Multiply and reduce a and b by prime 2^64-2^32+1
  const uint64_t P = 18446744069414584321;

  // Multiply
  #ifdef __CUDA_ARCH__
  uint64_t cHi = __umul64hi(a,b);
  uint64_t cLo = a*b;


  // Reduce
  uint64_t x3 = (cHi >> 32);
  uint64_t x2 = (cHi & UINT32_MAX);
  uint64_t x1 = (cLo >> 32);
  uint64_t x0 = (cLo & UINT32_MAX);

  uint64_t X1 = (x1<<32);
  uint64_t X2 = (x2<<32);


  uint64_t res = (X1+X2+x0-x2-x3);
  bool testA = (((x2+x3) > X1+X2+x0) && !((X1+X2 < X1) ||  ((X1+X2)+x0 < x0)));
  bool testB = ((res >= P) );
  bool testC = ((X1+X2 < X1) ||  ((X1+X2)+x0 < x0));

  // This avoids conditional branchs
  res = (P-res)*(testA) + (res-P)*(!testA && testB) + (res+4294967295L)*(!testA && !testB && testC) + (res)*(!testA && !testB && !testC);


  // if(((x2+x3) > X1+X2+x0) && !((X1+X2 < X1) ||  ((X1+X2)+x0 < x0))) {
  //   // printf("Negative Overflow!\n");
  //   res = P - res;
  //  }else if ((res >= P) ){
  //   // printf(" Overflow!\n");
  //   res -= P;
  //  }else if((X1+X2 < X1) ||  ((X1+X2)+x0 < x0)){
  //   // printf(" Double Overflow\n");
  //   res += 4294967295L;
  // }
   #else
  uint64_t res = (((__uint128_t)a) * ((__uint128_t)b) )%P;
  // __uint128_t c = ((__uint128_t)a) * ((__uint128_t)b);
  // uint64_t cHi = (c>>64);
  // uint64_t cLo = (c & UINT64_MAX);
  #endif
  return res;
}

__device__ __host__  uint64_t s_add(volatile uint64_t a,volatile uint64_t b){
  // Multiply and reduce a and b by prime 2^64-2^32+1
  uint64_t res = a+b;
  res += (res < a)*4294967295L;
  res = s_rem(res);
  return res;
}

__host__ __device__ void butterfly(uint64_t *v){
	// Butterfly
  uint64_t v0 = v[0];
  v[0] = s_add(v0,v[1]);
  v[1] = v0 - v[1];
}

__host__ __device__ int expand(int idxL, int N1, int N2){
	return (idxL/N1)*N1*N2 + (idxL%N1);
}

__device__ __host__ void NTTIteration(cuyasheint_t *W,
                                      cuyasheint_t *WInv,
                                      const int i,
                                      const int j,
                                      const int N,
                                      const int R,
                                      const int Ns,
                                      const cuyasheint_t* data0,
                                      cuyasheint_t *data1,
                                      const int type){
	uint64_t v[2];
	int idxS = j+i;
	// int wIndex;
	cuyasheint_t *w;
	if(type == FORWARD)
		w = W;
	else
		w = WInv;

	for(int r=0; r<R; r++){
    v[r] = s_mul(data0[idxS+r*N/R],w[i]);
  }

	butterfly(v);
	int idxD = expand(j,Ns,R);
	for(int r=0; r<R;r++){

    if(type == FORWARD)
  		data1[idxD+r*Ns] = v[r];
    else
      data1[idxD+r*Ns] = v[r];
  }
}

__global__ void NTT(cuyasheint_t *d_W,cuyasheint_t *d_WInv,const int N, const int R, const int Ns, cuyasheint_t* dataI, cuyasheint_t* dataO,const int type){

  const int ntt_index = (blockIdx.x)*N;
  for(int i = 0; i < N/R; i += 1024){
    // " Threads virtuais "
    const int j = (threadIdx.x+i);
    if( j < N*gridDim.x)
      NTTIteration(d_W,d_WInv,ntt_index,j, N, R, Ns, dataI, dataO,type);
  }
}

__global__ void polynomialNTTMul(cuyasheint_t *a,const cuyasheint_t *b,const int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      uint64_t a_value = a[tid];
      uint64_t b_value = b[tid];

      // In-place
      a[tid] = s_mul(a_value,b_value);
      // a[tid] = a_value*b_value % 18446744069414584321;
  }
}
#endif

__global__ void polynomialOPInteger(int opcode,cuyasheint_t *a,cuyasheint_t b,int size){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < size )
      // Coalesced access to global memory. Doing this way we reduce required bandwich.

    switch(opcode)
    {
    case ADD:
      if(tid == 0)
        a[tid] += b;
    case SUB:
      if(tid == 0)
        a[tid] -= b;
    case DIV:
        a[tid] /= b;
    case MUL:
        a[tid] *= b;
    case MOD:
        a[tid] %= b;
    }

}

__host__ void CUDAFunctions::callPolynomialOPInteger(int opcode,cudaStream_t stream,cuyasheint_t *a,cuyasheint_t b,int N,int NPolis)
{
  // This method just multiply all elements of array a and store the result inplace
  const int size = N*NPolis;

  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomialOPInteger<<< gridDim,blockDim, 1, stream>>> (opcode,a,b,N*NPolis);
  assert(cudaGetLastError() == cudaSuccess);

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
    polynomialPlainMul<<<gridDim,blockDim>>>(a,b,d_result,N,NPolis);
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
    result = cudaMemset(d_a,0,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_b,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    result = cudaMemset(d_b,0,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

    const int RADIX = 2;
  	dim3 blockDim(std::min(N/RADIX,1024));
  	dim3 gridDim(NPolis);

    // Forward
    for(int Ns=1; Ns<N; Ns*=RADIX){
      NTT<<<gridDim,blockDim,1,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,RADIX,Ns,a,d_a,FORWARD);
      assert(cudaGetLastError() == cudaSuccess);
      std::swap(a,d_a);
    }
    std::swap(a,d_a);

    for(int Ns=1; Ns<N; Ns*=RADIX){
      NTT<<<gridDim,blockDim,1,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,RADIX,Ns,b,d_b,FORWARD);
      assert(cudaGetLastError() == cudaSuccess);
      std::swap(b,d_b);
    }
    // Multiply
    dim3 blockDimMul(ADDBLOCKXDIM);
    dim3 gridDimMul((size)/ADDBLOCKXDIM+1); // We expect that ADDBLOCKXDIM always divide size
    polynomialNTTMul<<<gridDimMul,blockDimMul>>>(d_a,d_b,size);
    assert(cudaGetLastError() == cudaSuccess);

    // // Inverse
    for(int Ns=1; Ns<N; Ns*=RADIX){
      NTT<<<gridDim,blockDim,1,stream >>>(CUDAFunctions::d_W,CUDAFunctions::d_WInv,N,RADIX,Ns,d_a,d_result,INVERSE);
      assert(cudaGetLastError() == cudaSuccess);
      std::swap(d_a,d_result);
    }
    // cudaFree(d_a);
    // cudaFree(d_b);
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
    fft_radix16<<<gridDim,blockDim>>>(a,d_result,N);
    assert(cudaGetLastError() == cudaSuccess);

    // Multiply
    // polynomialFFTMul<<<gridDim,blockDim>>>(d_A,d_B,N*NPolis);

    // Inverse
    // fft_radix16<<<gridDim,blockDim>>>(d_A,d_result,N,NPolis,INVERSE);
    // assert(cudaGetLastError() == cudaSuccess);

    cudaFree(d_A);
    cudaFree(d_B);

  #elif defined(CUFFTMUL)
    const int size = N*NPolis;
    int size_c = N;
    int signal_size = N;
    Complex *d_a;
    Complex *d_b;
    Complex *d_c;
    cudaError_t result;

    result = cudaMalloc((void**)&d_result,size*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

    result = cudaMalloc((void **)&d_a, size*sizeof(Complex));
    assert(result == cudaSuccess);

    result = cudaMalloc((void **)&d_b, size*sizeof(Complex));
    assert(result == cudaSuccess);

    result = cudaMalloc((void **)&d_c, size*sizeof(Complex));
    assert(result == cudaSuccess);

    dim3 blockDim(32);
    dim3 gridDim(size/32 + (size % 32 == 0? 0:1));
    copyIntegerToComplex<<< gridDim,blockDim >>>(d_a,a,size);
    assert(cudaGetLastError() == cudaSuccess);

    copyIntegerToComplex<<< gridDim,blockDim >>>(d_b,b,size);
    assert(cudaGetLastError() == cudaSuccess);

    cufftResult fftResult;
    for(int i = 0; i < NPolis; i ++){
      int phase = N*i;
      // Apply FFT
      fftResult = cufftExecZ2Z(CUDAFunctions::plan, (cufftDoubleComplex *)(d_a+phase), (cufftDoubleComplex *)(d_a+phase), CUFFT_FORWARD);
      assert(fftResult == CUFFT_SUCCESS);
      // std::cout << "cufftExecZ2Z: "<< fftResult << std::endl;

      fftResult = cufftExecZ2Z(CUDAFunctions::plan, (cufftDoubleComplex *)(d_b+phase), (cufftDoubleComplex *)(d_b+phase), CUFFT_FORWARD);
      assert(fftResult == CUFFT_SUCCESS);
      // std::cout << "cufftExecZ2Z: "<< fftResult << std::endl;

    }

    polynomialcuFFTMul<<<gridDim,blockDim>>>(d_a,d_b,d_c,size_c,size);
    assert(cudaGetLastError() == cudaSuccess);


    for(int i = 0; i < NPolis; i ++){
      int phase = N*i;
      //getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
      // Apply inverse FFT
      fftResult = cufftExecZ2Z(CUDAFunctions::plan, (cufftDoubleComplex *)(d_c+phase), (cufftDoubleComplex *)(d_c+phase), CUFFT_INVERSE);
      assert(fftResult == CUFFT_SUCCESS);
      // std::cout << "cufftExecZ2Z: "<< fftResult << std::endl;
    }
    copyAndNormalizeComplexRealPartToInteger<<< gridDim,blockDim >>>(d_result,d_c,size,1.0f/signal_size);
    assert(cudaGetLastError() == cudaSuccess);


  //Destroy CUFFT context
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);


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

  ZZ PZZ = conv<ZZ>("18446744069414584321");
  cuyasheint_t k = conv<cuyasheint_t>(PZZ-1)/N;
  ZZ wNZZ = NTL::PowerMod(ZZ(7),k,PZZ);
  assert((PZZ-1)%(N) == 0);
  // const cuyasheint_t k = (P-1)/N;
  wN = conv<cuyasheint_t>(wNZZ);
  std::cout << wN << std::endl;;

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
  #elif defined(CUFFTMUL)
    cufftResult fftResult;
    fftResult = cufftPlan1d(&CUDAFunctions::plan, N, CUFFT_Z2Z, 1);
    assert(fftResult == CUFFT_SUCCESS);

  #endif
}

__global__ void polynomialReduction(cuyasheint_t *a,const int half,const int N,const int NPolis){     
  // This kernel must have N*Npolis/2 threads

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int cid = tid & (N/2-1); // We suppose that N = 2^k
  const int residueID = tid*2 / N; 

  if(2*tid < N*NPolis){
    a[residueID*N + cid] -= a[residueID*N + cid + half];
    __syncthreads();
    a[residueID*N + cid + half] = 0;
  }
}

void Polynomial::reduce(){
  // Just like DivRem, but here we reduce a with a cyclotomic polynomial
  Polynomial phi = this->get_phi();
  if(!this->get_device_updated()){
    #ifdef VERBOSE
    std::cout << "Reduce on host." << std::endl;
    #endif
    Polynomial quot;
    Polynomial rem;
    Polynomial::DivRem((*this),phi,quot, rem);
    this->copy(rem);
    return;
  }else{

    #ifdef VERBOSE
    std::cout << "Reduce on device." << std::endl;
    #endif

    const int half = phi.deg()/2;
    const int N = this->CRTSPACING;
    const int NPolis = this->CRTPrimes.size();
    const int size = N*NPolis/2;

    dim3 blockDim(32);
    dim3 gridDim(size/32 + (size % 32 == 0? 0:1));

    polynomialReduction<<<gridDim,blockDim>>>( this->get_device_crt_residues(),
                                                half,
                                                N,
                                                NPolis);
    cudaError_t result = cudaGetLastError();
    assert(result == cudaSuccess);
  }
}