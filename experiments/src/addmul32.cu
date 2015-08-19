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
#define NITERATIONS 300
#define FIRSTITERATION 256
#define LASTITERATION 4096
#define ADDBLOCKXDIM 32

double compute_time_ms(struct timespec start,struct timespec stop){
  return (( stop.tv_sec - start.tv_sec )*BILLION + ( stop.tv_nsec - start.tv_nsec ))/MILLION;
}

typedef uint32_t inteiro;

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
      atomicAdd((inteiro*)(&(a_hat[i])),(inteiro)(value));
      __syncthreads();
    }
  }

}

__global__ void NTT2(inteiro *W,inteiro *a, inteiro *a_hat, int N,int NPolis){
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
    inteiro value;
    // In each iteration, computes a_hat[i]
    for(int i = 0; i < N; i++){
      value += W[i + cid*N]*a[i];
    }
    a_hat[cid] = value;
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
	for(int degree = FIRSTITERATION; degree <= LASTITERATION; degree *= 2){
		// MUL
		std::cout << "###################################" << std::endl;

		const inteiro P = 4294955009;
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

		std:: cout << "4 kernels" << std::endl;
		clock_gettime( CLOCK_REALTIME, &start);
		for(int i = 0; i < NITERATIONS;i++){
  		NTT<<<gridDim,blockDim>>>(d_W,d_a,d_A,N,1); // Forward
  		NTT<<<gridDim,blockDim>>>(d_W,d_b,d_B,N,1); // Forward
			polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
  		NTT<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse


			result = cudaDeviceSynchronize();
			assert(result == cudaSuccess);
		}
		clock_gettime( CLOCK_REALTIME, &stop);
		std::cout << degree <<") NTT1 Mul kernel 32 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

	    std:: cout << "4 kernels" << std::endl;
	    clock_gettime( CLOCK_REALTIME, &start);
	    for(int i = 0; i < NITERATIONS;i++){
	      NTT2<<<gridDim,blockDim>>>(d_W,d_a,d_A,N,1); // Forward
	      NTT2<<<gridDim,blockDim>>>(d_W,d_b,d_B,N,1); // Forward
	      polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
	      NTT2<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse


	      result = cudaDeviceSynchronize();
	      assert(result == cudaSuccess);
	    }
	    clock_gettime( CLOCK_REALTIME, &stop);
	    std::cout << degree <<") NTT2 Mul kernel 32 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

	    std:: cout << "4 kernels" << std::endl;
	    clock_gettime( CLOCK_REALTIME, &start);
	    for(int i = 0; i < NITERATIONS;i++){
	      NTT3<<<gridDim,blockDim>>>(d_W,d_a,d_A,N,1); // Forward
	      NTT3<<<gridDim,blockDim>>>(d_W,d_b,d_B,N,1); // Forward
	      polynomialNTTMul<<<gridDim,blockDim>>>(d_A,d_B,d_C,N);
	      NTT3<<<gridDim,blockDim>>>(d_WInv,d_c,d_C,N,1);// Inverse


	      result = cudaDeviceSynchronize();
	      assert(result == cudaSuccess);
	    }
	    clock_gettime( CLOCK_REALTIME, &stop);
	    std::cout << degree <<") NTT3 Mul kernel 32 bits: " << compute_time_ms(start,stop)/NITERATIONS << std::endl<< std::endl;

	}
}