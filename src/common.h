#ifndef COMMON_H
#define COMMON_H
#define ADDBLOCKXDIM 32
#define CRTPRIMESIZE 31
// #define VERBOSE
// #define PLAINMUL
#define NTTMUL
// #define FFTMUL
//typedef uint32_t cuyasheint_t;
typedef uint64_t cuyasheint_t;
enum add_mode_t {ADD,SUB};
enum ntt_mode_t {INVERSE,FORWARD};

#endif
