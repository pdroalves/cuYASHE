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
#define NITERATIONS 100
#define ADDBLOCKXDIM 32

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

__device__ void sumReduce(inteiro value,inteiro *a,int i,inteiro q,int N, int NPolis);

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
__global__ void polynomialHalfNTTMul(const inteiro *a,inteiro *A,const inteiro *b,inteiro *B,inteiro *c,inteiro *C,const inteiro *W, const inteiro *WInv,const int N){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  inteiro a_value;
  inteiro b_value;

  // A NTT
  if(tid < N){

    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      int cid = tid % N; // Coefficient id

      sumReduce(W[cid + i*N]*A[cid],c,i,1,N,1);
      __syncthreads();
    }
  }

   // B NTT
  if(tid < N){

    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      int cid = tid % N; // Coefficient id

      sumReduce(W[cid + i*N]*B[cid],c,i,1,N,1);
      __syncthreads();
    }
  }

  // Mul
  if(tid < N ){
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = A[tid];
      b_value = B[tid];

      a_value *= b_value;

      C[tid] = a_value;
  }
  // __syncthreads();

}

__global__ void polynomialFullNTTMul(const inteiro *a,inteiro *A,const inteiro *b,inteiro *B,inteiro *c,inteiro *C,const inteiro *W, const inteiro *WInv,const int N){
  // We have one thread per polynomial coefficient on 32 threads-block.
  // For CRT polynomial adding, all representations should be concatenated aligned
  const int tid = threadIdx.x + blockDim.x*blockIdx.x;
  inteiro a_value;
  inteiro b_value;

  // A NTT
  if(tid < N){

    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      int cid = tid % N; // Coefficient id

      // sumReduce(W[cid + i*N]*A[cid],c,i,1,N,1);

      inteiro value = W[cid + i*N]*a[cid];
      // sumReduce(W[cid + i*N]*a[cid],a_hat,i,1,N,NPolis);
      atomicAdd((unsigned long long*)(&(A[i])),(unsigned long long)(value));
      // __syncthreads();
    }

   // B NTT

    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      int cid = tid % N; // Coefficient id

      // sumReduce(W[cid + i*N]*B[cid],c,i,1,N,1);
	  inteiro value = W[cid + i*N]*B[cid];
      atomicAdd((unsigned long long*)(&(B[i])),(unsigned long long)(value));
    }
      __syncthreads();

  // Mul
      // Coalesced access to global memory. Doing this way we reduce required bandwich.
      a_value = A[tid];
      b_value = B[tid];

      a_value *= b_value;

      C[tid] = a_value;
  __syncthreads();

  // INTT

    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      int cid = tid % N; // Coefficient id

      // sumReduce(WInv[cid + i*N]*C[cid],c,i,1,N,1);
      inteiro value = WInv[cid + i*N]*C[cid];
      // sumReduce(W[cid + i*N]*a[cid],a_hat,i,1,N,NPolis);
      atomicAdd((unsigned long long*)(&(c[i])),(unsigned long long)(value));
    }
  }

}


__device__ void sumReduce(inteiro value,inteiro *a,int i,inteiro q,int N, int NPolis){
  // Sum all elements in array "r" and writes to a, in position i

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  __shared__ inteiro r[ADDBLOCKXDIM];
  r[threadIdx.x] = value;

  if(tid < N*NPolis){

    // int stage = blockDim.x;
    // while(stage > 0){// Equivalent to for(int i = 0; i < lrint(log2(N))+1;i++)
    //   if(threadIdx.x < stage/2){
    //     // Only half of the threads are used
    //     r[threadIdx.x] += r[threadIdx.x + stage/2];
    //   	__syncthreads();
    //   }
    //   stage /= 2;
    // }
    atomicAdd((unsigned long long*)(&(r[0])),(unsigned long long)(r[threadIdx.x]));
    __syncthreads();

    // After this loop, r[0] hold the sum of all block data

    if(threadIdx.x == 0)
      // atomicAdd((unsigned long long*)(&(a[i])),(unsigned long long)(r[threadIdx.x]));
      a[i] = r[threadIdx.x];
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
      // atomicAdd((unsigned long long*)(&(a_hat[i])),(unsigned long long)(value));
      __syncthreads();
    }
  }

}


__device__ void sumReduce_tiled(inteiro value,inteiro *a,int i,inteiro q,int N, int NPolis){
  // Sum all elements in array "r" and writes to a, in position i

  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  __shared__ inteiro r[ADDBLOCKXDIM][ADDBLOCKXDIM];
  r[threadIdx.y][threadIdx.x] = value;

  if(tid < N*NPolis){

    int stage = blockDim.x;
    while(stage > 0){// Equivalent to for(int i = 0; i < lrint(log2(N))+1;i++)
      if(threadIdx.x < stage/2 && (tid % N) + stage/2 < N){
        // Only half of the threads are used
        r[threadIdx.y][threadIdx.x] += r[threadIdx.y][threadIdx.x + stage/2];
      	__syncthreads();
      }
      stage /= 2;
    }
    // After this loop, r[0] hold the sum of all block data

    if(threadIdx.x == 0)
      // atomicAdd((unsigned long long*)(&(a[i])),(unsigned long long)(r[threadIdx.x]));
      atomicAdd((unsigned long long*)(&(a[i])),(unsigned long long)(r[threadIdx.y][0]));
      // a[i] = r[threadIdx.y][0];
  }
}
__global__ void NTT_tiled(inteiro *W,inteiro *a, inteiro *a_hat, int N,int NPolis){
	// This algorithm supposes that N is power of 2, divisible by 32
	// We expect square blocks 32x32
	// N threads per dimension
	// Input:
	// w: Matrix of wNs
	// a: residues
	// a_hat: output
	// N: # of coefficients of each polynomial
	// NPolis: # of residues
	const int TILEDIM = blockDim.x;

	// blockDim.x == blockDim.y
	const int tidX = threadIdx.x + blockIdx.x*blockDim.x;
	const int tidY = threadIdx.y + blockIdx.y*blockDim.y;
	const int cid = tidX % N;

	for(int tileY = 0;tileY < N/TILEDIM; tileY++)
		for(int tileX = 0;tileX < N/TILEDIM; tileX++){

			inteiro value = W[cid + (threadIdx.y + tileY*TILEDIM)*N]*a[cid];

			sumReduce_tiled(W[cid + (threadIdx.y + tileY*TILEDIM)*N]*a[cid],a_hat,threadIdx.y + tileY*TILEDIM,1,N,NPolis);
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
	// 	// ADD

	// 	inteiro* h_a;
	// 	inteiro* h_b;
	// 	inteiro* d_a;
	// 	inteiro* d_b;
	// 	inteiro* d_c;
	// 	int N = degree+1;
	// 	struct timespec start, stop;

	// 	h_a = (inteiro*)malloc(N*sizeof(inteiro));
	// 	h_b = (inteiro*)malloc(N*sizeof(inteiro));

	// 	for(int i = 0; i < N;i++){
	// 		h_a[i] = rand();
	// 		h_b[i] = rand();
	// 	}
	// 	cudaError_t result;
	// 	clock_gettime( CLOCK_REALTIME, &start);
	// 	for(int i = 0; i < NITERATIONS;i++){

	// 		result = cudaMalloc((void**)&d_a,N*sizeof(inteiro));
	// 		assert(result == cudaSuccess);
	// 		result = cudaMalloc((void**)&d_b,N*sizeof(inteiro));
	// 		assert(result == cudaSuccess);
	// 		result = cudaMalloc((void**)&d_c,N*sizeof(inteiro));
	// 		assert(result == cudaSuccess);

	// 		result = cudaMemcpy(d_a,h_a , N*sizeof(inteiro), cudaMemcpyHostToDevice);
	// 		assert(result == cudaSuccess);
	// 		result = cudaMemcpy(d_b,h_b , N*sizeof(inteiro), cudaMemcpyHostToDevice);
	// 		assert(result == cudaSuccess);
	// 		result = cudaDeviceSynchronize();
	// 		assert(result == cudaSuccess);
	// 	}
	// 	clock_gettime( CLOCK_REALTIME, &stop);
	// 	std::cout << degree <<") Alloc and copy time 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl;

	// 	dim3 blockDim(ADDBLOCKXDIM);
	// 	dim3 gridDim(N/ADDBLOCKXDIM);

	// 	clock_gettime( CLOCK_REALTIME, &start);
	// 	for(int i = 0; i < NITERATIONS;i++){
	// 		polynomialAdd<<<gridDim,blockDim>>>(d_a,d_b,d_c,N);
	// 		result = cudaDeviceSynchronize();
	// 		assert(result == cudaSuccess);
	// 	}
	// 	clock_gettime( CLOCK_REALTIME, &stop);
	// 	std::cout << degree <<") Add kernel 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl;

	// }


	std::cout << "Mul: "<< std::endl << std::endl;
	for(int degree = 4096; degree <= 4096; degree *= 2){
		// MUL
		std::cout << "###################################" << std::endl;

		const inteiro P = 0xffffffff00000001;
		assert((P-1)%(degree) == 0);
		const inteiro k = (P-1)/(degree);
		const inteiro wN = powerMod(7,k,P);
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
		std::cout << degree <<") Alloc and copy time 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl;

		dim3 blockDim(ADDBLOCKXDIM);
		dim3 gridDim(N/ADDBLOCKXDIM);

		std:: cout << "4 kernels" << std::endl;
		clock_gettime( CLOCK_REALTIME, &start);
		cudaStream_t streamA;
    	cudaStreamCreate(&streamA);
		cudaStream_t streamB;
    	cudaStreamCreate(&streamB);

		for(int i = 0; i < NITERATIONS;i++){
  			NTT<<<gridDim,blockDim,1,streamA>>>(d_W,d_a,d_A,N,1); // Forward
  			NTT<<<gridDim,blockDim,1,streamB>>>(d_W,d_b,d_B,N,1); // Forward
			result = cudaDeviceSynchronize();
			assert(result == cudaSuccess);
			polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
  			NTT<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse

			// polynomialFullNTTMul<<<gridDim,blockDim>>>(d_a,d_A,d_b,d_B,d_c,d_C,d_W,d_WInv,N);
			// polynomialHalfNTTMul<<<gridDim,blockDim>>>(d_a,d_A,d_b,d_B,d_c,d_C,d_W,d_WInv,N);

  	// 		NTT<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse

			result = cudaDeviceSynchronize();
			assert(result == cudaSuccess);
		}
		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << degree <<") Mul kernel 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

		// std:: cout << "4 tiled kernels" << std::endl;
		// dim3 tiledBlockDim(ADDBLOCKXDIM,ADDBLOCKXDIM);
		// dim3 tiledGridDim(N/ADDBLOCKXDIM,N/ADDBLOCKXDIM);
		// clock_gettime( CLOCK_REALTIME, &start);
		// for(int i = 0; i < NITERATIONS;i++){
  // 			NTT_tiled<<<tiledGridDim,tiledBlockDim>>>(d_W,d_a,d_A,N,1); // Forward
  // 			NTT_tiled<<<tiledGridDim,tiledBlockDim>>>(d_W,d_b,d_B,N,1); // Forward
		// 	polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
  // 			NTT_tiled<<<tiledGridDim,tiledBlockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse

		// 	// polynomialFullNTTMul<<<gridDim,blockDim>>>(d_a,d_A,d_b,d_B,d_c,d_C,d_W,d_WInv,N);
		// 	// polynomialHalfNTTMul<<<gridDim,blockDim>>>(d_a,d_A,d_b,d_B,d_c,d_C,d_W,d_WInv,N);

  // 	// 		NTT<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse

		// 	result = cudaDeviceSynchronize();
		// 	assert(result == cudaSuccess);
		// }
		// clock_gettime( CLOCK_REALTIME, &stop);
		// std::cout << degree <<") Mul kernel 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

		// std:: cout << "2 kernels" << std::endl;
		// clock_gettime( CLOCK_REALTIME, &start);
		// for(int i = 0; i < NITERATIONS;i++){
  // 	// 		NTT<<<gridDim,blockDim>>>(d_W,d_a,d_A,N,1); // Forward
  // 	// 		NTT<<<gridDim,blockDim>>>(d_W,d_b,d_B,N,1); // Forward
		// 	// polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
  // 	// 		NTT<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse

		// 	// polynomialFullNTTMul<<<gridDim,blockDim>>>(d_a,d_A,d_b,d_B,d_c,d_C,d_W,d_WInv,N);
		// 	polynomialHalfNTTMul<<<gridDim,blockDim>>>(d_a,d_A,d_b,d_B,d_c,d_C,d_W,d_WInv,N);

  // 			NTT<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse

		// 	result = cudaDeviceSynchronize();
		// 	assert(result == cudaSuccess);
		// }
		// clock_gettime( CLOCK_REALTIME, &stop);
		// std::cout << degree <<") Mul kernel 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

		// std:: cout << "1 kernel" << std::endl;
		// clock_gettime( CLOCK_REALTIME, &start);
		// for(int i = 0; i < NITERATIONS;i++){
  // 	// 		NTT<<<gridDim,blockDim>>>(d_W,d_a,d_A,N,1); // Forward
  // 	// 		NTT<<<gridDim,blockDim>>>(d_W,d_b,d_B,N,1); // Forward
		// 	// polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
  // 	// 		NTT<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse

		// 	polynomialFullNTTMul<<<gridDim,blockDim>>>(d_a,d_A,d_b,d_B,d_c,d_C,d_W,d_WInv,N);
		// 	// polynomialHalfNTTMul<<<gridDim,blockDim>>>(d_a,d_A,d_b,d_B,d_c,d_C,d_W,d_WInv,N);

  // 	// 		NTT<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse

		// 	result = cudaDeviceSynchronize();
		// 	assert(result == cudaSuccess);
		// }
		// clock_gettime( CLOCK_REALTIME, &stop);
		// std::cout << degree <<") Mul kernel 64 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

	}
}