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


  if(this->aftermul)
    this->convert();
  if(b.aftermul)
    b.convert();

  /**
   * At this point the function expects that [c1]_q and [c2]_q
   */

  Polynomial g = common_multiplication<Polynomial>(this,&b);
  // g.reduce();
  g = Yashe::t*g;

  Ciphertext p;
  p.update_crt_spacing(g.get_crt_spacing());
  g.icrt();
  callCiphertextMulAux( p.d_bn_coefs, 
                        g.d_bn_coefs, 
                        Yashe::q, 
                        g.deg()+1, 
                        get_stream());
  
  // p.aftermul = true;
  // p.level = std::max(this->level,b.level)+1;
  // p.set_crt_computed(false);
  // p.set_icrt_computed(true);
  // p.set_host_updated(false);

  // g.release();
  return p;

}

void Ciphertext::convert(){
    this->keyswitch();
    this->aftermul = false;
    return;
}


template<int WORDLENGTH>
void worddecomp(Ciphertext *c, std::vector<Polynomial> *P){
  for(int i = 0; i <= c->deg() ; i++){
    ZZ x = c->get_coeff(i);
    int j = 0;
    while(x > 0){
      Polynomial p = P->at(j);
      p.set_coeff(i,(x & Yashe::WDMasking));
      x = NTL::RightShift(x,WORDLENGTH);
      j++;
    }

  }
}

template<>
void worddecomp<32>(Ciphertext *c, std::vector<Polynomial> *P){
  #pragma omp parallel for
  for(int i = 0; i <= c->deg() ; i++){
    ZZ x = c->get_coeff(i);
    int j = 0;
    while(x > 0){
      Polynomial p = P->at(j);
      p.set_coeff(i,conv<uint32_t>(x));
      x = NTL::RightShift(x,32);
      j++;
    }

  }
}


void Ciphertext::keyswitch(){
  #ifdef CYCLECOUNTING
  uint64_t start,end;
  start = get_cycles();
  #endif

  std::vector<Polynomial> P(Yashe::lwq,2*Polynomial::global_phi->deg()-1);

  /**
   * On Device
   */
  update_device_data();
  if(Yashe::w == to_ZZ("4294967296")){
    // Not used
    bn_t W;
    bn_t u_W;
    //
    
    callWordDecomp<32>( &P,
                        this->d_bn_coefs,
                        Yashe::lwq,
                        deg()+1,
                        W,
                        u_W,
                        get_stream()
                      );    
  }else{
    throw "Unknown Yashe::word";
  }
  // std::cout << Yashe::w << std::endl;
  // get_words(&W,Yashe::w);
  // bn_t u_W = get_reciprocal(Yashe::w);

   /**
    * On Host
    */
  // if(Yashe::w == to_ZZ("4294967296")) 
  //   worddecomp<32>(this,&P);
  // else
  //   throw "Unknown Yashe::word";
  
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

