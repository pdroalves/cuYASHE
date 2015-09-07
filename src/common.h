#ifndef COMMON_H
#define COMMON_H

// #define PLAINMUL
// #define NTTMUL
// #define FFTMUL
#define CUFFTMUL

#define ADDBLOCKXDIM 32
#ifdef CUFFTMUL
#define CRTPRIMESIZE 9
#else
#define CRTPRIMESIZE 31
#endif
// #define VERBOSE
//typedef uint32_t cuyasheint_t;
typedef uint64_t cuyasheint_t;
enum add_mode_t {ADD,SUB};
enum ntt_mode_t {INVERSE,FORWARD};

#endif
