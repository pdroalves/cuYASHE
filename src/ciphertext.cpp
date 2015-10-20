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

uint64_t cycles() {
  unsigned int hi, lo;
  asm (
    "cpuid\n\t"/*serialize*/
    "rdtsc\n\t"/*read the clock*/
    "mov %%edx, %0\n\t"
    "mov %%eax, %1\n\t" 
    : "=r" (hi), "=r" (lo):: "%rax", "%rbx", "%rcx", "%rdx"
  );
  return ((uint64_t) lo) | (((uint64_t) hi) << 32);
}


Ciphertext Ciphertext::operator*(Ciphertext b){
  // uint64_t start,end;

  // start = cycles();

  Ciphertext c1(*this);
  Ciphertext c2(b);

  if(c1.aftermul)
    c1.convert();
  if(c2.aftermul)
    c2.convert();

  #ifdef VERBOSE
  // std::cout << "c1 " << c1.to_string() << std::endl;
  // std::cout << "c2 " << c2.to_string() << std::endl;
  // std::cout << "Yashe::t " << Yashe::t << std::endl;
  #endif

  Polynomial g = common_multiplication<Polynomial>(&c1,&c2);
  g.icrt();
  g *= (Yashe::t);
  g.reduce();
  // Ciphertext p = g / Yashe::q;
  Ciphertext p;
  for(int i = 0; i <= g.deg();i++){
    ZZ quot;
    ZZ diff;
    NTL::DivRem(quot,diff,g.get_coeff(i),Yashe::q);

    quot %= Yashe::q;
    diff %= Yashe::q;

    if(2*diff > Yashe::q)
      p.set_coeff(i,(quot+1));
    else
      p.set_coeff(i,quot);
  } 

  p.aftermul = true;
  p.level = std::max(this->level,b.level)+1;
  p.set_device_updated(false);
  p.set_host_updated(true);
  // end = cycles();
  // std::cout << "ciphertext mult " << (end-start) << std::endl;

  return p;

}

void Ciphertext::convert(){
    this->keyswitch();
    this->aftermul = false;
    return;
}

void Ciphertext::keyswitch(){

  std::vector<Polynomial> P(Yashe::lwq);
  this->worddecomp(&P);
  this->set_coeffs();//Discards all coefficients

  for(int i = 0; i < Yashe::lwq; i ++){
    Polynomial p = (P[i]);
    *this += p*(Yashe::gamma[i]);
  }
  this->reduce();

}

void Ciphertext::worddecomp(std::vector<Polynomial> *P){

  for(int i = 0; i <= this->deg(); i++){
    ZZ c = this->get_coeff(i);
    int j = 0;
    while(c > 0){
      Polynomial p = P->at(j);
      p.set_coeff(i,(c & Yashe::WDMasking));
      c = NTL::RightShift(c,conv<long>(Yashe::w));
      j++;
    }
  }
}
