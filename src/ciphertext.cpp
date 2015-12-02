#include "ciphertext.h"
#include "yashe.h"

Ciphertext Ciphertext::operator+(Ciphertext &b){
  Ciphertext c = common_addition<Ciphertext>(this,&b);
  c.level = std::max(this->level,b.level);

  return c;
}

Ciphertext Ciphertext::operator+(Polynomial &b){
  Polynomial p = common_addition<Polynomial>(this,&b);
  Ciphertext c(p);
  return c;
}
Ciphertext Ciphertext::operator+=(Ciphertext &b){
  common_addition_inplace<Ciphertext>(this,&b);
  return *this;
}
Ciphertext Ciphertext::operator+=(Polynomial &b){
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


Ciphertext Ciphertext::operator*(Ciphertext &b){
  uint64_t start,end;
  uint64_t start_total,end_total;

  start_total = get_cycles();
  Ciphertext c1(*this);
  Ciphertext c2(b);

  start = cycles();
  if(c1.aftermul)
    c1.convert();
  if(c2.aftermul)
    c2.convert();
  end = cycles();
  // std::cout << "ciphertext convert " << (end-start) << std::endl;

  #ifdef VERBOSE
  std::cout << "ciphertext mult " << std::endl;
  
  #endif

  Polynomial g = common_multiplication<Polynomial>(&c1,&c2);
  g *= (Yashe::t.get_value());
  g.reduce();

  Ciphertext p;
  p.set_coeffs(g.deg()+1);
  end = cycles();

  start = cycles();
  for(int i = 0; i <= g.deg();i++){
    ZZ quot = g.get_coeff(i)/Yashe::q;
    ZZ diff = g.get_coeff(i)%Yashe::q;
    // NTL::DivRem(quot,diff,g.get_coeff(i),Yashe::q);

    if(2*diff > Yashe::q)
      p.set_coeff(i,(quot+1) % Yashe::q);
    else
      p.set_coeff(i,quot % Yashe::q);
  } 
  end = cycles();
  // std::cout << "ciphertext mult loop " << (end-start) << std::endl;

  p.aftermul = true;
  p.level = std::max(this->level,b.level)+1;
  p.set_device_updated(false);
  p.set_host_updated(true);

  end_total = get_cycles();
  // std::cout << "ciphertext mult " << (end_total-start_total) << std::endl;

  return p;

}

void Ciphertext::convert(){
    this->keyswitch();
    this->aftermul = false;
    return;
}

void Ciphertext::keyswitch(){
  #ifdef CYCLECOUNTING
  uint64_t start,end;
  start = get_cycles();
  #endif

  this->update_host_data();

  std::vector<Polynomial> P(Yashe::lwq);
  for(int i = 0; i < Yashe::lwq;i++)
    P[i].set_coeffs(this->get_crt_spacing());
  this->worddecomp(&P);
  this->set_coeffs();//Discards all coefficients

  
  for(int i = 0; i < Yashe::lwq; i ++){
    P[i].normalize();
    Polynomial p = (P[i])*(Yashe::gamma[i]);
    *this += p;
  }
  this->reduce();

  #ifdef CYCLECOUNTING
  end = get_cycles();
  // std::cout << (end-start) << " cycles for the loop on keyswitch" << std::endl;
  #endif
}

void Ciphertext::worddecomp(std::vector<Polynomial> *P){
  #ifdef CYCLECOUNTING
  uint64_t start,end;
  start = get_cycles();
  #endif

  for(int i = 0; i <= this->deg() ; i++){
    ZZ c = this->get_coeff(i);
    int j = 0;
    while(c > 0){
      Polynomial p = P->at(j);
      p.set_coeff(i,(c & Yashe::WDMasking));
      c = NTL::RightShift(c,conv<long>(Yashe::w));
      j++;
    }

  }
  #ifdef CYCLECOUNTING
  end = get_cycles();
  std::cout << (end-start) << " cycles to worddecomp" << std::endl;
  #endif
}
