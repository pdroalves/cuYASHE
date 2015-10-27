
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

  delta = (q/t.get_value()); // q/t

  Polynomial g = this->xkey.get_sample(phi.deg()-1);
  g.reduce();
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
    fl.reduce();
    fl %= q;
    // Polynomial fl;
    // fl.set_coeff(0,1);
    // fl.set_coeff(3,1);

    f = t*fl + 1;

    // std::cout << "phi " << this->phi << std::endl;
    f.reduce();
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
      // fInv = f;
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

  h = t*fInv;
  h *= g;
  h.reduce();
  // std::cout << "h " << h.to_string() << std::endl;
  h %= q;

  gamma.resize(lwq);
  for(int k = 0 ; k < lwq; k ++){
    gamma[k] = Polynomial(f);//Copy

    for(int j = 0; j < k;j ++){
      gamma[k] *= w;
      gamma[k].reduce();
    }

    Polynomial e = xerr.get_sample(phi.deg()-1);
    e.reduce();
    e %= q;
    Polynomial s = xerr.get_sample(phi.deg()-1);
    s.reduce();
    e %= q;

    Polynomial hs = h*s;
    gamma[k] += e;
    gamma[k] += hs;
    gamma[k].reduce();
    gamma[k] %= q;
    gamma[k].update_device_data();
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
}

Ciphertext Yashe::encrypt(Polynomial m){

  // ZZ delta;
  #ifdef DEBUG
  std::cout << "delta: "<< delta.get_value() <<std::endl;
  #endif

  uint64_t start,end;

  start = get_cycles();
  Polynomial ps = xerr.get_sample(phi.deg()-1);
  Polynomial e = xerr.get_sample(phi.deg()-1);
  end = get_cycles();
  uint64_t gen_samples = end-start;

  #ifdef DEBUG
  std::cout << "ps: "<< ps <<std::endl;
  #endif
  #ifdef DEBUG
  std::cout << "e: "<< e <<std::endl;
  #endif

  start = get_cycles();
  Polynomial p;
  uint64_t build_p = get_cycles()-start;
  // std::cout << "h: " << h.to_string() << std::endl;
  // std::cout << "ps: " << ps.to_string() << std::endl;
  
  start = get_cycles();
  p = (h*ps);
  uint64_t hMps = get_cycles()-start;
  
  start = get_cycles();
  p += e;
  uint64_t addInplacee = get_cycles()-start;

  start = get_cycles();
  Polynomial mdelta = delta*m;
  uint64_t build_mdelta = get_cycles()-start;  
  
  start = get_cycles();
  p += mdelta;
  uint64_t addInplaceMdelta = get_cycles()-start;

  start = get_cycles();
  p.reduce();
  uint64_t p_reduce = get_cycles()-start;
  
  // p %= q;
  // uint64_t pModq = get_cycles()-start;

  std::cout << "gen_samples: " << gen_samples << std::endl; 
  std::cout << "build_p: " << build_p << std::endl; 
  std::cout << "hMps: " << hMps << std::endl; 
  std::cout << "addInplacee: " << addInplacee << std::endl; 
  std::cout << "build_mdelta: " << build_mdelta << std::endl; 
  std::cout << "addInplaceMdelta: " << addInplaceMdelta << std::endl; 
  std::cout << "p_reduce: " << p_reduce << std::endl; 
  // std::cout << "pModq: " << pModq << std::endl; 

  // std::cout << "phi.deg() " << phi.deg() << std::endl;
  // std::cout << "h " << h.to_string() << std::endl;
  // std::cout << "ps " << ps.to_string() << std::endl;
  // std::cout << "e " << e.to_string() << std::endl;
  // std::cout << "mdelta " << mdelta.to_string() << std::endl;
  // std::cout << "p " << p.to_string() << std::endl;

  #ifdef DEBUG
  std::cout << "ciphertext: "<< p <<std::endl;
  #endif
  Ciphertext c(p);

  // delete &ps;
  // delete &e;
  // delete &mdelta;

  return c;
}
Polynomial Yashe::decrypt(Ciphertext c){
  #ifdef VERBOSE
  std::cout << "Yashe decrypt" << std::endl;
  #endif
  // std::cout << "f " << f.to_string() << std::endl;
  // std::cout << "c " << c.to_string() << std::endl;

  Polynomial g;
  if(c.aftermul){
    #ifdef VERBOSE
    std::cout << "aftermul" << std::endl;
    #endif
    g = f*f;
    g.reduce();
    g %= Yashe::q;
    g *= c;
    
    // std::cout << "f*f:" << g.to_string() << std::endl;
    // std::cout << "f*f*c:" << g.to_string() << std::endl;

  }else{
    #ifdef VERBOSE
    std::cout << "not  aftermul" << std::endl;
    #endif
    g = f*c;
  }
  g.reduce();
  g %= Yashe::q;

  ZZ coeff = g.get_coeff(0)*t.get_value();
  ZZ quot;
  ZZ rem;
  NTL::DivRem(quot,rem,coeff,q);

  quot %= q;
  rem %= q;
  #ifdef VERBOSE
  std::cout << "rem: " << rem << std::endl;
  std::cout << "coeff: " << coeff << std::endl;
  std::cout << "q: " << q << std::endl;
  std::cout << "2*rem: " << 2*rem << std::endl;
  #endif
  
  Polynomial m;
  if(2*rem > q){
    if(coeff == t.get_value()-1){
      m.set_coeff(0,0);
    }else{
      m.set_coeff(0,(quot+1)%t.get_value());
    }
  }else{
    m.set_coeff(0,(quot)%t.get_value());
  }

  return m;
}