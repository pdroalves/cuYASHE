#ifndef COMMON_H
#define COMMON_H
#define ADDBLOCKXDIM 32
#define CRTPRIMESIZE 10
// #define VERBOSE
// #define PLAINMUL
#define NTTMUL
// #define FFTMUL
enum add_mode_t {ADD,SUB};
enum ntt_mode_t {INVERSE,FORWARD};

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif
