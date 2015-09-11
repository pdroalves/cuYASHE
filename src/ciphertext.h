#ifndef CIPHERTEXT_H
#define CIPHERTEXT_H

#include "polynomial.h"

class Ciphertext: public Polynomial{
  public:
    Ciphertext operator+(Ciphertext b);
    Ciphertext operator+(Polynomial b);
    Ciphertext operator+=(Ciphertext b){
      this->set_device_crt_residues( ((*this)+b).get_device_crt_residues());
      return *this;
    }
    Ciphertext operator+=(Polynomial b){
      this->set_device_crt_residues( ((*this)+b).get_device_crt_residues());
      return *this;
    }
    Ciphertext operator*(Ciphertext b);
    void convert();
    Ciphertext operator=(Polynomial p){
      level = 0;
      this->copy(p);
      return *this;
    }
    Ciphertext(Polynomial *p){
        level = 0;
        this->copy(*p);
    }
    Ciphertext(Polynomial p){
        level = 0;
        this->copy(p);
    }
    Ciphertext(){
      level = 0;
    }

    void copy(Polynomial p){
        level = 0;
        Polynomial::copy(p);
    }

    int level = 0;
    bool aftermul = false;

  private:
    void keyswitch();
    void worddecomp(std::vector<Polynomial> *P);
};

#endif
