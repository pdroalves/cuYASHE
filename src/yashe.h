#ifndef YASHE_H
#define YASHE_H

#include <NTL/ZZ.h>
#include "polynomial.h"
#include "ciphertext.h"
#include "distribution.h"

class Yashe{
  private:
    Distribution xkey;
    Distribution xerr;

  public:
    static int d;
    static Polynomial phi;
    static ZZ q;
    static uint64_t t;
    static ZZ w;
    static std::vector<Polynomial> gamma;
    static Polynomial h;
    static Polynomial f;
    static int lwq;
    static ZZ WDMasking;
    Yashe(){
      const int sigma_err = 8;
      const float gaussian_std_deviation = sigma_err*0.4;
      const int gaussian_bound = sigma_err*6;
      xkey = Distribution(NARROW);
      xerr = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);
    };
    Yashe(float gaussian_std_deviation, int gaussian_bound){
      xkey = Distribution(NARROW);
      xerr = Distribution(DISCRETE_GAUSSIAN,gaussian_std_deviation, gaussian_bound);
    };
    void generate_keys();
    Ciphertext encrypt(Polynomial m);
    Polynomial decrypt(Ciphertext c);
};

#endif
