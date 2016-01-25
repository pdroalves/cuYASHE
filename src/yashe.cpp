
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
    f %= q;

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
  ff %= q;

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
    gamma[k] %= q;
    gamma[k].update_device_data();

    #ifdef DEBUG
    std::cout << "e = " << e.to_string() << std::endl;
    std::cout << "s = " << s.to_string() << std::endl;
    std::cout << "gamma[" << k << "] = " << gamma[k].to_string() << std::endl;
    #endif
  }

  // Word decomp mask
  WDMasking = NTL::LeftShift(ZZ(1),conv<long>(Yashe::w - 1));


  #ifdef VERBOSE
  std::cout << "Keys generated." << std::endl;
  #endif
}

Ciphertext Yashe::encrypt(Polynomial m){

  #ifdef DEBUG
  std::cout << "delta: "<< delta.get_value() <<std::endl;
  #endif
  struct timespec start,stop;
  clock_gettime( CLOCK_REALTIME, &start);
  /** 
   * ps will be used in a D degree multiplication, resulting in a 2*D degree polynomial
   * e will be used in a 2*D degree addition
   */
  Polynomial ps = xerr.get_sample(phi.deg()-1,2*(phi.deg()-1));
  Polynomial e = xerr.get_sample(phi.deg()-1,2*(phi.deg()-1));
  clock_gettime( CLOCK_REALTIME, &stop);
  float diff = compute_time_ms(start,stop);
  // std::cout << "sampling: " << diff << std::endl;

  #ifdef DEBUG
  std::cout << "ps: "<< ps.to_string() <<std::endl;
  std::cout << "e: "<< e.to_string() <<std::endl;
  #endif
 
  Polynomial p;
  p = (h*ps) + e;
  Polynomial mdelta = delta*m;
  p += mdelta;
  p.reduce();


  // std::cout << "phi.deg() " << phi.deg() << std::endl;
  // std::cout << "h " << h.to_string() << std::endl;
  // std::cout << "ps " << ps.to_string() << std::endl;
  // std::cout << "e " << e.to_string() << std::endl;
  // std::cout << "mdelta " << mdelta.to_string() << std::endl;
  // std::cout << "p " << p.to_string() << std::endl;

  Ciphertext c(p);
  return c;
}
Polynomial Yashe::decrypt(Ciphertext c){
  #ifdef VERBOSE
  std::cout << "Yashe decrypt" << std::endl;
  #endif
  // std::cout << "f " << f.to_string() << std::endl;
  // std::cout << "c " << c.to_string() << std::endl;
  // uint64_t start,end;


  Polynomial g;

  if(c.aftermul){
    #ifdef VERBOSE
    std::cout << "aftermul" << std::endl;
    #endif
    g = ff*c;    
    // std::cout << "f*f:" << g.to_string() << std::endl;
    // std::cout << "f*f*c:" << g.to_string() << std::endl;

  }else{
    #ifdef VERBOSE
    std::cout << "not  aftermul" << std::endl;
    #endif
    // f.set_crt_residues_computed(false);
    g = f*c;

  }
  g.reduce();
  g %= q;
  ZZ g_value = g.get_coeff(0);

  ZZ coeff = g_value*t.get_value();
  ZZ quot;
  ZZ rem;
  NTL::DivRem(quot,rem,coeff,q);

  quot %= q;
  #ifdef VERBOSE
  std::cout << "g_value: " << g_value << std::endl;
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

  // std::cout << (end-start) << " cycles to decrypt I" << std::endl;
  return m;
}