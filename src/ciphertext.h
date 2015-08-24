#ifndef CIPHERTEXT_H
#define CIPHERTEXT_H

#include "polynomial.h"

class Ciphertext: public Polynomial{
  public:
    Ciphertext operator+(Ciphertext b);
    Ciphertext operator*(Ciphertext b);
    void convert();
    Ciphertext operator=(Polynomial p){
      level = 0;
      this->copy(p);

      this->set_device_crt_residues(p.get_device_crt_residues());
      this->set_host_updated(p.get_host_updated());
      this->set_device_updated(p.get_device_updated());

      return *this;
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

    int level;
    bool aftermul = false;

  private:
    void keyswitch();
    void worddecomp(std::vector<Polynomial> *P);
};

#endif
