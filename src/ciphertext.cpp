#include "ciphertext.h"
#include "yashe.h"

Ciphertext Ciphertext::operator+(Ciphertext b){
  Ciphertext c = common_addition<Ciphertext>(this,&b);
  c.level = std::max(this->level,b.level);

  return c;
}

Ciphertext Ciphertext::operator+(Polynomial b){
  Polynomial p = common_addition<Polynomial>(this,&b);
  Ciphertext c(p);
  return c;
}
Ciphertext Ciphertext::operator+=(Ciphertext b){
  common_addition_inplace<Ciphertext>(this,&b);
  return *this;
}
Ciphertext Ciphertext::operator+=(Polynomial b){
  common_addition_inplace<Polynomial>(this,&b);
  return *this;
}

Ciphertext Ciphertext::operator*(Ciphertext b){

  Ciphertext c1(*this);
  Ciphertext c2(b);

  if(c1.aftermul)
    c1.convert();
  if(c2.aftermul)
    c2.convert();

  #ifdef DEBUG
  std::cout << "c1 " << c1 << std::endl;
  std::cout << "c2 " << c2 << std::endl;
  std::cout << "Yashe::t " << Yashe::t << std::endl;
  #endif

  Polynomial g = common_multiplication<Polynomial>(&c1,&c2)*(Yashe::t);
  g.reduce();
  for(int i = 0; i <= g.deg();i++){
    ZZ quot;
    ZZ diff;
    NTL::DivRem(quot,diff,g.get_coeff(i),Yashe::q);

    if(2*diff > Yashe::q)
      this->set_coeff(i,(quot+1) % Yashe::q);
    else
      this->set_coeff(i,quot % Yashe::q);
  }

  this->aftermul = true;
  this->level = std::max(this->level,b.level)+1;
  this->set_device_updated(false);
  this->set_host_updated(true);
  return *this;

}

void Ciphertext::convert(){
    this->keyswitch();
    this->aftermul = false;
    return;

}

void Ciphertext::keyswitch(){

      std::vector<Polynomial> P(Yashe::lwq);
      this->worddecomp(&P);
      this->set_coeffs(std::vector<cuyasheint_t>());//Discards all coefficients

      for(int i = 0; i < Yashe::lwq; i ++){
        Polynomial p = (P[i]);
        *this += p*(Yashe::gamma[i]);
      }
      this->reduce();
}

void Ciphertext::worddecomp(std::vector<Polynomial> *P){
      //
      // Todo
      // std::cout << "worddecomp implemented for ciphertext." << std::endl;
      // throw exception();

      ZZ MASKING = NTL::LeftShift(ZZ(1),conv<long>(Yashe::w - 1));

      for(int i = 0; i <= this->deg(); i++){
        ZZ c = this->get_coeff(i);
        int j = 0;
        while(c > 0){
          Polynomial p = P->at(j);
          p.set_coeff(i,(c&MASKING));
          c = NTL::RightShift(c,conv<long>(Yashe::w));
          j++;
        }
      }

}
