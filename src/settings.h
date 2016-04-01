/**
 * cuYASHE
 * Copyright (C) 2015-2016 cuYASHE Authors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef SETTINGS_H
#define SETTINGS_H

#include <cstdint>

// This define the default transform for polynomial multiplication
// #define NTTMUL_TRANSFORM
#define CUFFTMUL_TRANSFORM

enum transforms {NTTMUL, CUFFTMUL};

#define ADDBLOCKXDIM 32
#ifdef CUFFTMUL_TRANSFORM
#define CRTPRIMESIZE 19 
#define PRIMES_BUCKET_SIZE 56                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     //
extern const uint32_t PRIMES_BUCKET[];
#else
#define CRTPRIMESIZE 10 
#define PRIMES_BUCKET_SIZE 200                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           //
extern const uint32_t PRIMES_BUCKET[];
#endif
// #define DEBUG
// #define VERBOSE
// #define VERBOSEMEMORYCOPY

/**
 * If defined, the program will print a message everytime it
 * applies CRT, ICRT, update_*_data or update_crt_spacing.
 */
// #define SPEEDCHECK

// CRT cannot use primes bigger than WORD/2 bits
#define WORD 64

// Standard number of words to allocate
// #define STD_BNT_WORDS_ALLOC 32 // Up to 1024 bits big integers
#define STD_BNT_WORDS_ALLOC 10 // Up to  bits big integers
#define DSTD_BNT_WORDS_ALLOC 20 // Up to  bits big integers

// We use cuyasheint_t as uint64_t to simplify operations
typedef uint64_t cuyasheint_t;
// typedef uint32_t cuyasheint_t;

enum add_mode_t {ADD,SUB,MUL,DIV,MOD};
enum ntt_mode_t {INVERSE,FORWARD};
// enum reduction_type {RESIDUES,COEFS};
#define RESIDUES 0
#define COEFS 1

#define PREDUCTION RESIDUES

#define asm	__asm__ volatile
#endif
