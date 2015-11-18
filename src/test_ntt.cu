#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "cuda_functions.h"
#include "settings.h"
#include "polynomial.h"

int main(){

  // Plan settings
  const int N = 8;
  const int batch = 2;

  CUDAFunctions::init(2*N);
  // Memory alloc
  uint64_t *h_a;
  uint64_t *h_b;
  uint64_t *d_a1;
  uint64_t *aux;
  uint64_t *d_a2;

  h_a = (uint64_t*)malloc(2*N*batch*sizeof(uint64_t));
  h_b = (uint64_t*)malloc(2*N*batch*sizeof(uint64_t));

  cudaError_t result = cudaMalloc((void**)&d_a1,2*N*batch*sizeof(uint64_t));
  assert(result == cudaSuccess);
  result = cudaMalloc((void**)&d_a2,2*N*batch*sizeof(uint64_t));
  assert(result == cudaSuccess);
  result = cudaMalloc((void**)&aux,2*N*batch*sizeof(uint64_t));
  assert(result == cudaSuccess);

  // Data load
  std::cout << "a: "<< std::endl;
  for(unsigned int pol = 0; pol < batch;pol++)
    for(unsigned int coeff = 0; coeff < 2*N; coeff++){
      std::cout << "a[" << coeff << "] = ";     
      if(coeff < N)
        h_a[pol*(2*N) + coeff] = 42*(pol+1);
      else
        h_a[pol*(2*N) + coeff] = 0;
      std::cout << h_a[pol*N + coeff] << std::endl;
    }
  // Memcpy
  result = cudaMemcpy(d_a1,h_a,2*N*batch*sizeof(uint64_t),cudaMemcpyHostToDevice);
  assert(result == cudaSuccess);

  // NTT
  result = cudaMemset((void*)aux,0,(2*N)*batch*sizeof(uint64_t));
  assert(result == cudaSuccess);
  CUDAFunctions::callNTT(2*N, batch, d_a1, aux,FORWARD);

  result = cudaMemcpy(d_a2,d_a1,2*N*batch*sizeof(uint64_t),cudaMemcpyDeviceToDevice);
  assert(result == cudaSuccess);

  result = cudaMemset((void*)aux,0,(2*N)*batch*sizeof(uint64_t));
  assert(result == cudaSuccess);
  CUDAFunctions::callNTT(2*N, batch,d_a2,aux,INVERSE);

  // Memcpy
  result = cudaMemcpy(h_b,d_a2,2*N*batch*sizeof(uint64_t),cudaMemcpyDeviceToHost);

  for(unsigned int pol = 0; pol < batch;pol++){
    std::cout << "residue " << pol << std::endl;
    for(unsigned int coeff = 0; coeff < 2*N; coeff++)
      std::cout << "h_a[" << coeff <<"]: " << h_a[pol*N + coeff] << " == " << "h_b[" << coeff <<"]: " << h_b[pol*N + coeff]/(2*N) << std::endl;;
  }

  free(h_a);
  cudaFree(d_a1);
  cudaFree(d_a2);
  cudaFree(aux);
}
