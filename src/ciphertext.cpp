#include "ciphertext.h"
#include "yashe.h"

Ciphertext Ciphertext::operator+(Ciphertext b){

  Polynomial poly_a(*this);
  Polynomial poly_b(b);

  Polynomial c = poly_a+poly_b;

  int new_level = max(this->level,b.level);

  this->copy(c);
  this->level = new_level;

  return *this;
}

Ciphertext Ciphertext::operator*(Ciphertext b){

  Ciphertext c1(*this);
  Ciphertext c2(b);

  if(c1.aftermul)
    c1.convert();
  if(c2.aftermul)
    c2.convert();

  Polynomial C1(c1);
  Polynomial C2(c2);

  #ifdef DEBUG
  std::cout << "C1 " << C1 << std::endl;
  std::cout << "C2 " << C2 << std::endl;
  std::cout << "Yashe::t " << Yashe::t << std::endl;
  #endif

  Polynomial g = C1*C2*(Yashe::t) % Yashe::phi;
  Polynomial pmult;
  for(int i = 0; i <= g.deg();i++){
    ZZ quot;
    ZZ diff;
    NTL::DivRem(quot,diff,g.get_coeff(i),Yashe::q);

    if(2*diff > Yashe::q)
      pmult.set_coeff(i,quot+1);
    else
      pmult.set_coeff(i,quot);
  }

  this->copy(pmult);
  (*this) %= Yashe::q;

  this->aftermul = true;

  this->level = std::max(this->level,b.level)+1;

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
      *this %= Yashe::phi;
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
