#include "ciphertext.h"
#include "yashe.h"
#include "cuda_ciphertext.h"

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

  Ciphertext c1(*this);
  Ciphertext c2(b);

  if(c1.aftermul)
    c1.convert();
  if(c2.aftermul)
    c2.convert();

  Polynomial g = common_multiplication<Polynomial>(&c1,&c2);
  g *= (Yashe::t.get_value());
  g.reduce();

  Ciphertext p;
  p.set_coeffs(g.deg()+1);

  for(int i = 0; i <= g.deg();i++){
    ZZ quot = g.get_coeff(i)/Yashe::q;
    ZZ diff = g.get_coeff(i)%Yashe::q;

    if(2*diff > Yashe::q)
      p.set_coeff(i,(quot+1) % Yashe::q);
    else
      p.set_coeff(i,quot % Yashe::q);
  } 

  p.aftermul = true;
  p.level = std::max(this->level,b.level)+1;
  p.set_crt_computed(false);
  p.set_icrt_computed(false);
  p.set_host_updated(true);

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

  std::vector<Polynomial> P(Yashe::lwq,2*Polynomial::global_phi->deg()-1);
  // this->worddecomp(&P);
  bn_t WDMasking;
  get_words(&WDMasking,Yashe::WDMasking);
  int deg = this->deg();
  update_device_data();
  callWordDecomp( &P,
                  this->d_bn_coefs,
                  deg,
                  Yashe::lwq,
                  NumBits(Yashe::w),
                  WDMasking,
                  get_stream());
  
  for(int i = 0; i < Yashe::lwq; i ++){
    Polynomial p = (P[i])*(Yashe::gamma[i]);
    *this += p;
  }
  this->reduce();

  #ifdef CYCLECOUNTING
  end = get_cycles();
  // std::cout << (end-start) << " cycles for the loop on keyswitch" << std::endl;
  #endif
}