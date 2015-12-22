#ifndef SETTINGS_H
#define SETTINGS_H

#include <cstdint>

// #define PLAINMUL
#define NTTMUL
// #define FFTMUL
// #define CUFFTMUL

#define ADDBLOCKXDIM 32
#ifdef CUFFTMUL
#define CRTPRIMESIZE 10
#else
#define CRTPRIMESIZE 29
#endif
// #define DEBUG
// #define CYCLECOUNTING
// #define VERBOSE
// #define VERBOSEMEMORYCOPY
// #define MAYADDONCPU

// CRT cannot use primes bigger than 32 bits
#define WORD 32
#define STD_BNT_ALLOC 32 // 1024 bits big integers

// We use cuyasheint_t as uint64_t to simplify operations
typedef uint64_t cuyasheint_t;

enum add_mode_t {ADD,SUB,MUL,DIV,MOD};
enum ntt_mode_t {INVERSE,FORWARD};

#define asm	__asm__ volatile

#endif