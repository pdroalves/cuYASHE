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
#define PRIMES_BUCKET_SIZE 200                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           //
extern const uint32_t PRIMES_BUCKET[];
#endif
// #define DEBUG
// #define CYCLECOUNTING
// #define VERBOSE
// #define VERBOSEMEMORYCOPY
// #define MAYADDONCPU


// CRT cannot use primes bigger than WORD/2 bits
#define WORD 64

// Standard number of words to allocate
#define STD_BNT_WORDS_ALLOC 32 // Up to 1024 bits big integers

// We use cuyasheint_t as uint64_t to simplify operations
typedef uint64_t cuyasheint_t;
// typedef uint32_t cuyasheint_t;

enum add_mode_t {ADD,SUB,MUL,DIV,MOD};
enum ntt_mode_t {INVERSE,FORWARD};
// enum reduction_type {RESIDUES,COEFS};
#define RESIDUES 0
#define COEFS 1

#define PREDUCTION COEFS

#define asm	__asm__ volatile

#endif