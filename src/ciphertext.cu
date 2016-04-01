/**
 * cuYASHE
 * Copyright (C) 2015-2016 cuYASHE Authors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
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
  Ciphertext g(common_multiplication<Polynomial>(this,&b));

  /**
   * This should not be necessary. 
   * Someone need to modify the Integer class to allow in-place multiplication
   */
  g.set_device_crt_residues( 
      CUDAFunctions::callPolynomialOPInteger( MUL,
        g.get_stream(),
        g.get_device_crt_residues(),
        Yashe::t,
        g.get_crt_spacing(),
        Polynomial::CRTPrimes.size()
      )
      );

  g.icrt(); 
  callCiphertextMulAux(g.d_bn_coefs, Yashe::q, g.deg(), g.get_stream());
  
  g.aftermul = true;
  g.level = std::max(this->level,b.level)+1;
  g.set_crt_computed(false);
  g.set_icrt_computed(true);
  g.set_host_updated(false);

  // g.release();
  return g;
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

// #define CYCLECOUNTING
void Ciphertext::keyswitch(){
  #ifdef CYCLECOUNTING
  uint64_t start,end;
  start = get_cycles();
  #endif

  /**
   * On Device
   */
  this->reduce();
  this->icrt();
  assert(Yashe::w == to_ZZ("4294967296"));
  #ifdef CYCLECOUNTING
  end = get_cycles();
  std::cout << (end-start) << " cycles for the loop on keyswitch" << std::endl;
  #endif


  callWordDecomp<32>( &Yashe::P,
                      this->d_bn_coefs,
                      Yashe::lwq,
                      deg()+1,
                      get_stream()
                    );    
  for(int i = 0; i < Yashe::lwq; i ++){
    assert(Yashe::P.at(i).get_crt_spacing() == Yashe::gamma[i].get_crt_spacing());
    int needed_spacing = Yashe::P.at(i).get_crt_spacing();

    // Optimization
    // Someone needs to refactor this!
    /////////
    // Mul //
    /////////
    #ifdef NTTMUL_TRANSFORM
    CUDAFunctions::callPolynomialMul( Yashe::P.at(i).get_device_crt_residues(),
                                      Yashe::P.at(i).get_device_crt_residues(),
                                      Yashe::gamma[i].get_device_crt_residues(),
                                      needed_spacing*Polynomial::CRTPrimes.size(),
                                      Yashe::P.at(i).get_stream()
                                      );
    #else
    CUDAFunctions::executeCuFFTPolynomialMul(   Yashe::P.at(i).get_device_transf_residues(), 
                                                Yashe::P.at(i).get_device_transf_residues(), 
                                                Yashe::gamma[i].get_device_transf_residues(), 
                                                needed_spacing*Polynomial::CRTPrimes.size(),
                                                Yashe::gamma[i].get_stream());
    #endif
    /////////
    // Add //
    /////////
    #ifdef NTTMUL_TRANSFORM
    CUDAFunctions::callPolynomialAddSubInPlace( this->get_stream(),
                                                this->get_device_crt_residues(),
                                                Yashe::P.at(i).get_device_crt_residues(),
                                                (int)(this->get_crt_spacing()*Polynomial::CRTPrimes.size()),
                                                ADD);
    #else
    CUDAFunctions::callPolynomialcuFFTAddSubInPlace(  this->get_stream(),
                                                      this->get_device_transf_residues(),
                                                      Yashe::P.at(i).get_device_transf_residues(),
                                                      (int)(this->get_crt_spacing()*Polynomial::CRTPrimes.size()),
                                                      ADD);
    #endif
  }

  this->reduce();

}


void Ciphertext::keyswitch_mul(std::vector<Polynomial> *P){
  // It is expected that Yashe::gamma[i] lies in NTT domain
  // 
  for(int i = 0; i < Yashe::lwq; i++){
    // NTT(P[i])
    P->at(i).crt();
    CUDAFunctions::applyNTT(P->at(i).get_device_crt_residues(),
                            P->at(i).get_crt_spacing(),
                            Polynomial::CRTPrimes.size(),
                            FORWARD,
                            get_stream());
    // P[i] *= Yashe::gamma[i];
    // *this += P[i];
    CUDAFunctions::executePolynomialMul(P->at(i).get_device_crt_residues(),
                                        P->at(i).get_device_crt_residues(),
                                        Yashe::gamma[i].get_device_crt_residues(),
                                        P->at(i).get_crt_spacing(),
                                        get_stream());
    CUDAFunctions::executePolynomialAdd(this->get_device_crt_residues(),
                                        this->get_device_crt_residues(),
                                        P->at(i).get_device_crt_residues(),
                                        P->at(i).get_crt_spacing(),
                                        get_stream());
  }
  CUDAFunctions::applyNTT(this->get_device_crt_residues(),
                          this->get_crt_spacing(),
                          Polynomial::CRTPrimes.size(),
                          INVERSE,
                          get_stream());
}