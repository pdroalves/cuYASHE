#ifndef SETTINGS_H
#define SETTINGS_H

// #define PLAINMUL
// #define NTTMUL
// #define FFTMUL
#define CUFFTMUL

#define ADDBLOCKXDIM 32
#ifdef CUFFTMUL
#define CRTPRIMESIZE 10
#else
#define CRTPRIMESIZE 31
#endif
// #define CYCLECOUNTING
// #define VERBOSE
// #define VERBOSEMEMORYCOPY
// #define MAYADDONCPU
// typedef uint32_t cuyasheint_t;
typedef uint64_t cuyasheint_t;
// typedef int64_t cuyasheint_t;
enum add_mode_t {ADD,SUB,MUL,DIV,MOD};
enum ntt_mode_t {INVERSE,FORWARD};

#define asm	__asm__ volatile

#endif