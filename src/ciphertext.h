#ifndef CIPHERTEXT_H
#define CIPHERTEXT_H

#include "polynomial.h"

class Ciphertext: public Polynomial{
  public:
    Ciphertext operator+(Ciphertext b);
    Ciphertext operator*(Ciphertext b);
    void convert();
    Ciphertext operator=(const ZZ_pEX p){
      level = 0;
      for(int i = 0; i < NTL::deg(p); i++)
        NTL::SetCoeff(*this,i,p[i]);
      return *this;
    }
    Ciphertext operator=(Polynomial p){
      level = 0;
      // Ciphertext new_p;
      for(int i = 0; i <= NTL::deg(p); i++)
        NTL::SetCoeff(*this,i,p[i]);

      this->set_cuda_device_pointer(p.getCudaDevicePointer());
      this->set_host_updated(p.HOST_IS_UPDATED);
      this->set_device_updated(p.DEVICE_IS_UPDATED);

      return *this;
    }
    Ciphertext(Polynomial p){
      level = 0;
      for(int i = 0; i <= NTL::deg(p); i++)
        NTL::SetCoeff(*this,i,p[i]);

      this->set_cuda_device_pointer(p.getCudaDevicePointer());
      this->set_host_updated(p.HOST_IS_UPDATED);
      this->set_device_updated(p.DEVICE_IS_UPDATED);

    }
    Ciphertext(){
      level = 0;
    }

    void copy(Polynomial p){
        level = 0;
        for(int i = 0; i <= NTL::deg(p); i++)
          NTL::SetCoeff(*this,i,p[i]);

        this->set_cuda_device_pointer(p.getCudaDevicePointer());
        this->set_host_updated(p.HOST_IS_UPDATED);
        this->set_device_updated(p.DEVICE_IS_UPDATED);
    }

    int level;
    bool aftermul = false;

  private:
    void keyswitch();
    void worddecomp(std::vector<Polynomial> *P);
};

#endif
