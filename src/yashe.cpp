#include "yashe.h"
#include "settings.h"
int Yashe::d = 0;
Polynomial Yashe::phi = Polynomial();
ZZ Yashe::q = ZZ(0);
uint64_t Yashe::t = 0;
ZZ Yashe::w = ZZ(0);
int Yashe::lwq = 0;
Polynomial Yashe::h = Polynomial();
std::vector<Polynomial> Yashe::gamma;
Polynomial Yashe::f = Polynomial();
ZZ Yashe::WDMasking = ZZ(0);
void Yashe::generate_keys(){
  #ifdef DEBUG
  std::cout << "generate_keys:" << std::endl;
  std::cout << "d: " << d << std::endl;
  std::cout << "phi: " << phi << std::endl;
  std::cout << "q: " << Polynomial::global_mod << std::endl;
  std::cout << "t: " << t << std::endl;
  std::cout << "w: " << w << std::endl;
  std::cout << "R: " << Polynomial::global_mod << std::endl;
  #endif

  Polynomial g = this->xkey.get_sample(phi.deg()-1);
  g %= phi;
  g %= q;
  // Polynomial g;
  // g.set_coeff(0,1);
  // g.set_coeff(1,1);
  // g.set_coeff(2,655615110);
  // g.set_coeff(3,1);

  #ifdef DEBUG
  std::cout << "g = " << g << std::endl;
  #endif
  // Computes a polynomial f with inverse
  Polynomial fInv;
  while(1==1){
    Polynomial fl = xkey.get_sample(phi.deg()-1);
    fl %= phi;
    fl %= q;
    // Polynomial fl;
    // fl.set_coeff(0,1);
    // fl.set_coeff(3,1);

    f = fl*t + 1;

    // std::cout << "phi " << this->phi << std::endl;
    f %= phi;
    f %= q;

    #ifdef DEBUG
    std::cout << "fl: " << fl << std::endl;
    std::cout << "f: " << f << std::endl;
    #endif
    try{
      #ifdef VERBOSE
      std::cout << "Computing invmod of f "<< std::endl;
      #endif
      fInv = Polynomial::InvMod(f,phi);
      //fInv = f;
      #ifdef VERBOSE
      std::cout << "Done." << std::endl;
      #endif

      break;
    } catch (exception& e)
    {
      #ifdef VERBOSE
      std::cout << "f has no modular inverse. " << e.what() << std::endl;
      #endif
    }
  }

  h = fInv*g;
  h *= t;
  h %= phi;
  h %= q;
  gamma.resize(lwq);
  for(int k = 0 ; k < lwq; k ++){
    gamma[k] = Polynomial(f);//Copy

    for(int j = 0; j < k;j ++){
      gamma[k] *= w;
      gamma[k] %= phi;
    }

    Polynomial e = xerr.get_sample(phi.deg()-1);
    e %= phi;
    e %= q;
    Polynomial s = xerr.get_sample(phi.deg()-1);
    s %= phi;
    e %= q;

    gamma[k] += e + h*s;
    gamma[k] %= phi;
    gamma[k] %= q;
    #ifdef DEBUG
    std::cout << "e = " << e << std::endl;
    std::cout << "s = " << s << std::endl;
    std::cout << "gamma[" << k << "] = " << gamma[k] << std::endl;
    #endif
  }

  // Word decomp mask
  WDMasking = NTL::LeftShift(ZZ(1),conv<long>(Yashe::w - 1));


  #ifdef VERBOSE
  std::cout << "Keys generated." << std::endl;
  #endif
  #ifdef DEBUG
  std::cout << "Keys generated." << std::endl;
  #endif
}

Ciphertext Yashe::encrypt(Polynomial m){

  // ZZ delta;
  ZZ delta = (q/t); // q/t
  #ifdef DEBUG
  std::cout << "delta: "<< delta <<std::endl;
  #endif

  Polynomial ps = xerr.get_sample(phi.deg()-1);
  ps %= phi;
  ps %= q;
  Polynomial e = xerr.get_sample(phi.deg()-1);
  e %= phi;
  e %= q;
  
  #ifdef DEBUG
  std::cout << "ps: "<< ps <<std::endl;
  #endif
  #ifdef DEBUG
  std::cout << "e: "<< e <<std::endl;
  #endif

  Polynomial p;

  p = (h*ps);
  p += e;
  p += m*delta;
  p %= phi;
  p %= q;

  #ifdef DEBUG
  std::cout << "ciphertext: "<< p <<std::endl;
  #endif

  Ciphertext c(p);
  return c;
}
Polynomial Yashe::decrypt(Ciphertext c){
  // std::cout << f.to_string() << std::endl;
  // std::cout << c.to_string() << std::endl;

  Polynomial g;
  if(c.aftermul){
    g = f*f*c;
  }else{
    g = f*c;
  }
  g.reduce();

  // Polynomial reduced_g = Polynomial::rem(g,phi);
  // reduced_g.icrt();
  #ifdef DEBUG
  std::cout << "g = f*c % phi: "<< reduced_g <<std::endl;
  #endif
  // std::cout << g.to_string() << std::endl;
  g *= t; // This operation should not be modular

  ZZ coeff = g.get_coeff(0);
  ZZ quot;
  ZZ rem;
  NTL::DivRem(quot,rem,coeff,q);

  #ifdef DEBUG
  std::cout << "quot: " << quot << std::endl;
  std::cout << "rem: " << rem << std::endl;
  std::cout << "coeff: " << coeff << std::endl;
  std::cout << "q: " << q << std::endl;
  std::cout << "2*rem: " << 2*rem << std::endl;
  #endif
  
  Polynomial m;
  if(2*rem > q){
    if(coeff == t-1){
      m.set_coeff(0,0);
    }else{
      m.set_coeff(0,(quot+1)%t);
    }
  }else{
    m.set_coeff(0,(quot)%t);
  }

  return m;
}