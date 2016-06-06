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
#include "yashe.h"
#include "settings.h"
int Yashe::d = 0;
Polynomial Yashe::phi = Polynomial();
ZZ Yashe::q = ZZ(0);
bn_t Yashe::qDiv2;
cuyasheint_t Yashe::t = 0;
ZZ Yashe::delta = to_ZZ(0);
ZZ Yashe::w = ZZ(0);
int Yashe::lwq = 0;
Polynomial Yashe::h = Polynomial();
std::vector<Polynomial> Yashe::gamma;
Polynomial Yashe::f = Polynomial();
Polynomial Yashe::ff = Polynomial();
Polynomial Yashe::tf = Polynomial();
Polynomial Yashe::tff = Polynomial();
ZZ Yashe::WDMasking = ZZ(0);
std::vector<Polynomial> Yashe::P;


void Yashe::generate_keys(){
  #ifdef DEBUG
  std::cout << "generate_keys:" << std::endl;
  std::cout << "d: " << d << std::endl;
  std::cout << "phi: " << phi.to_string() << std::endl;
  std::cout << "q: " << Polynomial::global_mod << std::endl;
  std::cout << "t: " << t.get_value() << std::endl;
  std::cout << "w: " << w << std::endl;
  std::cout << "R: " << Polynomial::global_mod << std::endl;
  #endif

  Polynomial g = this->xkey.get_sample(phi.deg()-1);
  #ifdef DEBUG
  std::cout << "g = " << g.to_string() << std::endl;
  #endif

  // Computes a polynomial f with inverse
  Polynomial fInv;
  while(1==1){
    Polynomial fl = xkey.get_sample(phi.deg()-1);

    f = fl*t + 1;
    f.reduce();

    // Overwrite f
    f.set_coeff(0,18);
    f.set_coeff(1,0);
    f.set_coeff(2,8174);
    f.set_coeff(3,8174);

    // #ifdef DEBUG
    std::cout << "fl: " << fl.to_string() << std::endl;
    std::cout << "f: " << f.to_string() << std::endl;
    // #endif
    try{
      fInv = Polynomial::InvMod(f,phi);
      fInv.normalize();
      // fInv = f;

      break;
    } catch (exception& e)
    {
      #ifdef VERBOSE
      std::cout << "f has no modular inverse. " << e.what() << std::endl;
      #endif
    }
  }

  // Pre-computed value
  ff = f*f;
  ff.reduce();

  tff = ff*t;
  tf = f*t;

  h = fInv*g*t;
  h.reduce();
  h.set_coeffs(4);
  h.set_coeff(0,415);
  h.set_coeff(1,5008);
  h.set_coeff(2,7081);
  h.set_coeff(3,7853);
  h.update_device_data();

  gamma.resize(lwq);
  // for(int k = 0 ; k < lwq; k ++){
  //   gamma[k] = Polynomial(f);//Copy

  //   for(int j = 0; j < k;j ++){
  //     gamma[k] *= w;
  //   }

  //   Polynomial e = xerr.get_sample(phi.deg()-1);
  //   Polynomial s = xerr.get_sample(phi.deg()-1);

  //   Polynomial hs = h*s;
  //   hs.reduce();
  //   gamma[k] += e;
  //   gamma[k] += hs;
  //   gamma[k].reduce();
  //   gamma[k].update_crt_spacing(2*(phi.deg()-1));
  //   gamma[k].update_device_data();

  //   #ifdef DEBUG
  //   std::cout << "e = " << e.to_string() << std::endl;
  //   std::cout << "s = " << s.to_string() << std::endl;
  //   std::cout << "gamma[" << k << "] = " << gamma[k].to_string() << std::endl;
  //   #endif
  // }

  // Word decomp mask
  WDMasking = NTL::LeftShift(ZZ(1),NumBits(Yashe::w))-1;

  //////////////////////////////////
  // Init static variables
  Yashe::ps.update_crt_spacing(2*phi.deg()-1);
  Yashe::e.update_crt_spacing(phi.deg()-1);
  //////////////////////////////////
  get_words(&qDiv2,q/2);
  //////////////////////////////////
  delta = (q/t); // q/t
  //////////////////////////////////
  bn_t *d_P;
  const int N = 2*Polynomial::global_phi->deg()-1;
  const int size = N*lwq;

  P.clear();
  P.resize(lwq,N);
  cudaError_t result = cudaMalloc((void**)&d_P,size*sizeof(bn_t));
  assert(result == cudaSuccess);

  bn_t *h_P;
  h_P = (bn_t*)malloc(size*sizeof(bn_t));

  // #pragma omp parallel for
  for(int i = 0; i < size; i++){
    h_P[i].dp = NULL;  
    get_words(&h_P[i],to_ZZ(0));
  }

  result = cudaMemcpy(d_P,h_P,size*sizeof(bn_t),cudaMemcpyHostToDevice);
  assert(result == cudaSuccess);

  for(int i = 0; i < lwq;i++){
    // cudaFree(P[i].d_bn_coefs);
    P[i].d_bn_coefs = d_P + i*N;
  }
  free(h_P);
  /////
  #ifdef VERBOSE
  std::cout << "Keys generated." << std::endl;
  #endif
}

Ciphertext Yashe::encrypt(Polynomial m){
  #ifdef DEBUG
  std::cout << "delta: "<< delta.get_value() <<std::endl;
  #endif
  /** 
   * ps will be used in a D degree multiplication, resulting in a 2*D degree polynomial
   * e will be used in a 2*D degree addition
   */
  // xerr.get_sample(&ps,phi.deg()-1);
  // xerr.get_sample(&e,phi.deg()-1);
  ps.set_coeffs(4);
  ps.set_coeff(0,8188);
  ps.set_coeff(1,0);
  ps.set_coeff(2,8190);
  ps.set_coeff(3,1);

  e.set_coeffs(4);
  e.set_coeff(0,2);
  e.set_coeff(1,1);
  e.set_coeff(2,0);
  e.set_coeff(3,8190);

  // #ifdef DEBUG
  std::cout << "h: "<< h.to_string() <<std::endl;
  std::cout << "ps: "<< ps.to_string() <<std::endl;
  std::cout << "e: "<< e.to_string() <<std::endl;
  // #endif
  
  Polynomial mdelta = m*delta;
  std::cout << "mdelta: "<< mdelta.to_string() <<std::endl;
  // ps *= h;
  // e += mdelta;
  // e += ps;
  assert(ps.get_crt_spacing() == h.get_crt_spacing());
  ps.update_device_data();
  h.update_device_data();
  /////////////////
  // ps = ps + h //
  /////////////////
  #ifdef NTTMUL_TRANSFORM
  CUDAFunctions::callPolynomialMul( ps.get_device_crt_residues(),
                                    ps.get_device_crt_residues(),
                                    h.get_device_crt_residues(),
                                    (int)(ps.get_crt_spacing())*Polynomial::CRTPrimes.size(),
                                    ps.get_stream()
                                    );
  #else
  CUDAFunctions::executeCuFFTPolynomialMul(   ps.get_device_transf_residues(), 
                                              ps.get_device_transf_residues(), 
                                              h.get_device_transf_residues(), 
                                              ps.get_crt_spacing()*Polynomial::CRTPrimes.size(),
                                              ps.get_stream());
  ps.set_crt_computed(false);
  ps.set_icrt_computed(false);
  #endif
  ps.set_host_updated(false);
  
  ps.reduce();

  /////////////////////
  // e = e + m*delta //
  // e = e + ps //
  ////////////////
  e.update_device_data();
  ps.update_device_data();
  mdelta.update_device_data();
  std::cout << "ps: "<< ps.to_string() <<std::endl;
  #ifdef NTTMUL_TRANSFORM

  CUDAFunctions::callPolynomialAddSubInPlace( e.get_stream(),
                                              e.get_device_crt_residues(),
                                              mdelta.get_device_crt_residues(),
                                              (int)(e.get_crt_spacing()*Polynomial::CRTPrimes.size()),
                                              ADD);

  CUDAFunctions::callPolynomialAddSubInPlace( e.get_stream(),
                                              e.get_device_crt_residues(),
                                              ps.get_device_crt_residues(),
                                              (int)(e.get_crt_spacing()*Polynomial::CRTPrimes.size()),
                                              ADD);
  #else
  CUDAFunctions::callPolynomialcuFFTAddSubInPlace(  e.get_stream(),
                                                    e.get_device_transf_residues(),
                                                    ps.get_device_transf_residues(),
                                                    (int)(e.get_crt_spacing()*Polynomial::CRTPrimes.size()),
                                                    ADD);
  CUDAFunctions::callPolynomialcuFFTAddSubInPlace(  e.get_stream(),
                                                    e.get_device_transf_residues(),
                                                    mdelta.get_device_transf_residues(),
                                                    (int)(e.get_crt_spacing()*Polynomial::CRTPrimes.size()),
                                                    ADD);
  e.set_crt_computed(false);
  e.set_icrt_computed(false);
  e.set_host_updated(false);
  std::cout << "e: "<< e.to_string() <<std::endl;
  #endif
  e.modn(q);

  Ciphertext c = e;
  return c;
}

Polynomial Yashe::decrypt(Ciphertext c){
  // #ifdef VERBOSE
  std::cout << std::endl << "Yashe decrypt" << std::endl;
  // #endif
  std::cout << "f " << f.to_string() << std::endl;
  std::cout << "c " << c.to_string() << std::endl;
  // uint64_t start,end;

  Polynomial m;

  if(c.aftermul){
    #ifdef VERBOSE
    std::cout << "aftermul" << std::endl;
    #endif
    m = ff*c;    
    // std::cout << "f*f:" << g.to_string() << std::endl;
    // std::cout << "f*f*c:" << g.to_string() << std::endl;

  }else{
    #ifdef VERBOSE
    std::cout << "not  aftermul" << std::endl;
    #endif
    // f.set_crt_residues_computed(false);
    m = f*c;
  }
  m.reduce();
  std::cout << "c*f:" << m.to_string() << std::endl;

  m = m * t;
  std::cout << "t*c*f:" << m.to_string() << std::endl;
  // m.update_device_data();
  // m.icrt();
  // callCiphertextMulAux(  m.d_bn_coefs, 
  //                       Yashe::q, 
  //                       m.get_crt_spacing(), 
  //                       m.get_stream());
  m.update_host_data();
  for(int i = 0; i <= m.deg(); i++){
    ZZ diff = m.get_coeff(i)%q;
    if(2*diff > q)
      m.set_coeff(i,m.get_coeff(i)/q +1);
    else
      m.set_coeff(i,m.get_coeff(i)/q);
  }
  m.set_transf_computed(false);
  m.set_itransf_computed(false);
  m.set_crt_computed(false);
  m.set_icrt_computed(false);
  m.set_host_updated(true);
  return m;
}