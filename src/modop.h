#ifndef MODOP_H
#define MODOP_H

typedef uint64_t u64;
typedef uint32_t u32;
#define MOD_P 0xffffffff00000001
__device__ u64 mod_sub(u64 x, u64 y)
{
    u64 r = x - y;
    /* if borrow generated - hopefully the compiler will optimise this! */
    if (x < y)
        r += MOD_P;        /* Add back p if borrow */
    return r;
}

__device__ u64 mod_add(u64 x, u64 y)
{
    x = MOD_P - x;        /* do addition by negating y then subtracting */
    u64 r = y - x;        /* y - (-x) */
    /* if borrow generated - hopefully the compiler will optimise this! */
    if (y < x)
        r += MOD_P;        /* Add back p if borrow */
    return r;
}
__device__ u64 mod_reduce(u64 b, u64 a)
{
    u32 d = b >>32,
        c = b;
    if (a >= MOD_P)                /* (x1, x0) */
        a -= MOD_P;
    a = mod_sub(a, c);
    a = mod_sub(a, d);
    a = mod_add(a, ((u64)c)<<32);
    return a;
}

__device__ u64 mod_mul(u64 x, u64 y)
{
    u32 a = x,
        b = x >>32,
        c = y,
        d = y >>32;

    /* first synthesise the product using 32*32 -> 64 bit multiplies */
    x = b * (u64)c;
    y = a * (u64)d;
    u64 e = a * (u64)c,
        f = b * (u64)d,
        t;

    x += y;                        /* b*c + a*d */
    /* carry? */
    if (x < y)
        f += 1LL << 32;            /* carry into upper 32 bits - can't overflow */

    t = x << 32;
    e += t;                        /* a*c + LSW(b*c + a*d) */
    /* carry? */
    if (e < t)
        f += 1;                    /* carry into upper 64 bits - can't overflow*/
    t = x >>32;
    f += t;                        /* b*d + MSW(b*c + a*d) */
    /* can't overflow */

    /* now reduce: (b*d + MSW(b*c + a*d), a*c + LSW(b*c + a*d)) */
    return mod_reduce(f, e);
}
#endif