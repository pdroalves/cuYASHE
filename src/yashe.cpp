
#include "yashe.h"
#include "settings.h"
int Yashe::d = 0;
Polynomial Yashe::phi = Polynomial();
ZZ Yashe::q = ZZ(0);
Integer Yashe::t = 0;
Integer Yashe::delta = 0;
ZZ Yashe::w = ZZ(0);
int Yashe::lwq = 0;
Polynomial Yashe::h = Polynomial();
std::vector<Polynomial> Yashe::gamma;
Polynomial Yashe::f = Polynomial();
Polynomial Yashe::ff = Polynomial();
ZZ Yashe::WDMasking = ZZ(0);

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

  delta = (q/t.get_value()); // q/t

  Polynomial g = this->xkey.get_sample(phi.deg()-1);
  #ifdef DEBUG
  std::cout << "g = " << g.to_string() << std::endl;
  #endif

  // Computes a polynomial f with inverse
  Polynomial fInv;
  while(1==1){
    Polynomial fl = xkey.get_sample(phi.deg()-1);

    f = t*fl + 1;
    f.reduce();

    #ifdef DEBUG
    std::cout << "fl: " << fl.to_string() << std::endl;
    std::cout << "f: " << f.to_string() << std::endl;
    #endif
    try{
      // fInv = Polynomial::InvMod(f,phi);
      // fInv.normalize();
      fInv = f;

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

  h = t*fInv*g;
  h.reduce();
  h.update_device_data();

  gamma.resize(lwq);
  for(int k = 0 ; k < lwq; k ++){
    gamma[k] = Polynomial(f);//Copy

    for(int j = 0; j < k;j ++){
      gamma[k] *= w;
    }

    Polynomial e = xerr.get_sample(phi.deg()-1);
    Polynomial s = xerr.get_sample(phi.deg()-1);

    Polynomial hs = h*s;
    hs.reduce();
    gamma[k] += e;
    gamma[k] += hs;
    gamma[k].reduce();
    gamma[k].update_crt_spacing((phi.deg()-1));
    gamma[k].update_device_data();

    #ifdef DEBUG
    std::cout << "e = " << e.to_string() << std::endl;
    std::cout << "s = " << s.to_string() << std::endl;
    std::cout << "gamma[" << k << "] = " << gamma[k].to_string() << std::endl;
    #endif
  }

  // Word decomp mask
  WDMasking = NTL::LeftShift(ZZ(1),NumBits(Yashe::w))-1;

  // Intialize the samples used by encrypt()
  Yashe::ps.update_crt_spacing(2*phi.deg()-1);
  Yashe::e.update_crt_spacing(phi.deg()-1);
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
  xerr.get_sample(&ps,phi.deg()-1);
  xerr.get_sample(&e,phi.deg()-1);

  #ifdef DEBUG
  std::cout << "ps: "<< ps.to_string() <<std::endl;
  std::cout << "e: "<< e.to_string() <<std::endl;
  #endif
 
  Polynomial mdelta = delta*m;
  ps *= h;
  e += mdelta;
  e += ps;
  // e.reduce();

  Ciphertext c = e;
  return c;
}

Polynomial Yashe::decrypt(Ciphertext c){
  #ifdef VERBOSE
  std::cout << "Yashe decrypt" << std::endl;
  #endif
  // std::cout << "f " << f.to_string() << std::endl;
  // std::cout << "c " << c.to_string() << std::endl;
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

  // m *= Yashe::t;
  m.icrt();
  callCiphertextMulAuxMersenne( m.d_bn_coefs, 
                                Yashe::q, 
                                m.get_crt_spacing(), 
                                m.get_stream());
  m.set_crt_computed(false);
  m.set_icrt_computed(true);
  m.set_host_updated(false);
  return m;
}