#include "polynomial.h"
#include "common.h"
#include <omp.h>
#include <assert.h>
#include <string.h>

ZZ Polynomial::CRTProduct = ZZ(1);
std::vector<cuyasheint_t> Polynomial::CRTPrimes(0);
ZZ Polynomial::global_mod = ZZ(0);
Polynomial *(Polynomial::global_phi) = NULL;
bool Polynomial::phi_set = false;

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

void Polynomial::update_device_data(unsigned int usable_ratio){

    if(this->get_device_updated())
      return;
    else if(!this->get_crt_computed())
      this->crt();

    #ifdef VERBOSE
    std::cout << "Copying data to GPU." << std::endl;
    #endif

    cudaError_t result;
    this->ON_COPY = true;

    // if(this->d_polyCRT == NULL){
      result = cudaMalloc((void**)&this->d_polyCRT,this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t));
      assert(result == cudaSuccess);
    // }

    // result = cudaMemset((void*)this->d_polyCRT,0,this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t));
    // #ifdef VERBOSE
    // std::cout << "cudaMemset:" << cudaGetErrorString(result) << std::endl;
    // #endif
    // assert(result == cudaSuccess);

    cuyasheint_t *aux;
    // aux = (cuyasheint_t*)malloc(this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t));
    aux = (cuyasheint_t*)calloc(this->CRTSPACING*(this->polyCRT.size()),sizeof(cuyasheint_t));
    for(unsigned int i=0;i < this->polyCRT.size();i++){
      memcpy(aux+this->CRTSPACING*i,&(this->polyCRT[i][0]),(this->polyCRT[i].size()/usable_ratio)*sizeof(cuyasheint_t));
    }

    result = cudaMemcpy(this->d_polyCRT, aux , this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    // result = cudaDeviceSynchronize();
    // assert(result == cudaSuccess);
    free(aux);
    this->ON_COPY = false;
    this->set_device_updated(true);
    assert(this->get_device_updated() == true);
}

void Polynomial::update_host_data(){
    if(this->get_host_updated())
      return;
    else if(!this->get_icrt_computed())
      this->icrt();

    #ifdef VERBOSE
    std::cout << "Copying data to CPU." << std::endl;
    #endif

    cudaError_t result;

    if(this->polyCRT.size() != Polynomial::CRTPrimes.size())
      this->polyCRT.resize(Polynomial::CRTPrimes.size());

    // Copy all data to host
    cuyasheint_t *aux;
    // aux = (cuyasheint_t*) calloc (this->polyCRT.size()*this->CRTSPACING,sizeof(cuyasheint_t));
    aux = (cuyasheint_t*) malloc (this->polyCRT.size()*this->CRTSPACING*sizeof(cuyasheint_t));
    result = cudaMemcpy(aux , this->d_polyCRT, this->polyCRT.size()*this->CRTSPACING*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
    //
    for(unsigned int i=0;i < this->polyCRT.size();i++){
      if(this->polyCRT[i].size() != (unsigned int)(this->CRTSPACING))
        this->polyCRT[i].resize(this->CRTSPACING);
        // *(this->polyCRT[i][0]) = *(aux[i*this->CRTSPACING]);
        // memcpy(&(this->polyCRT[i])[this->polyCRT[i][0].size() - this->CRTSPACING], &aux[i*this->CRTSPACING],  this->CRTSPACING * sizeof(cuyasheint_t));
        // std::copy(&(aux) + i*this->CRTSPACING,&(aux) + (i+1)*this->CRTSPACING,this->polyCRT[i][0]);
        for(unsigned int j=0; j < (unsigned int)(this->CRTSPACING);j++){
          this->polyCRT[i][j] = aux[j+i*this->CRTSPACING];
        }
    }
    free(aux);
    this->set_host_updated(true);
}

void Polynomial::crt(){
    // "The product of those prime numbers should be larger than the potentially largest coefficient of polynomial c, that we will obtain as a result of a computation for accurate recovery through ICRT." produtorio_{i=1}^n (pi) > n*q^2

    // if(this->CRTProduct == NULL or this->CRTPrimes == NULL){
    //     throw -1;
    // }

    // Escapes, if possible

    
    if(this->get_crt_computed())
      return;
    else if(!this->get_host_updated())
      this->update_host_data();
    
    std::vector<cuyasheint_t> P = this->CRTPrimes;
    this->polyCRT.resize(P.size());

    // Extract the coefficients to a array of ZZs
    std::vector<ZZ> array = this->get_coeffs();

    // We pick each prime
    for(std::vector<cuyasheint_t>::iterator iter_prime = P.begin(); iter_prime != P.end(); iter_prime++){
        int index = iter_prime - P.begin();//Debug

        // Apply mod at each coefficient
        std::vector<cuyasheint_t> rep = this->polyCRT[index];
        rep.resize(array.size());
        for(std::vector<ZZ>::iterator iter = array.begin();iter != array.end();iter++){
          // std::cout << "Prime: " << *iter_prime << std::endl;
          int array_i = iter-array.begin();//Debug
          rep[array_i] = conv<cuyasheint_t>(*iter % (*iter_prime));
          // std::cout << "rep : " << rep[array_i] << ", ";
        }

         polyCRT[index] = (rep);
    }

    this->set_host_updated(true);
    this->set_device_updated(false);
    this->set_crt_computed(true);
    this->set_icrt_computed(true);
}

void Polynomial::icrt(){
  // Escapes, if possible
  if(this->get_icrt_computed())
    return;
  else if(!this->get_host_updated()){
    this->set_icrt_computed(true);//If we do not set this, we get a infinite loop
    this->update_host_data();
  }

  std::vector<cuyasheint_t> P = this->CRTPrimes;
  ZZ M = this->CRTProduct;

  // Backup original modulus
  ZZ q = ZZ_p::modulus();
  ZZ_p::init(M);

  uint64_t start = cycles();
  // Convert all CRT representations to Polynomial
  std::vector<ZZ_pEX> X(this->polyCRT.size(),ZZ_pEX());
  for(int index = 0; index < X.size(); index ++){
      std::vector<cuyasheint_t> v = (this->polyCRT).at(index);

      ZZ_pEX z(v.size());
      for(std::vector<cuyasheint_t>::iterator iter = v.begin(); iter != v.end(); iter++){
        unsigned int index = iter-v.begin();
        NTL::SetCoeff(z,index,conv<ZZ_p>(*iter));
      }
      X[index] = z;
  }
  uint64_t end = cycles();
  // std::cout << "ZZ_pEX convertion: " << (end-start) << std::endl;

  // Computes each summation value
  start = cycles();
  std::vector<ZZ_pEX> icrti(P.size());
  for(std::vector<cuyasheint_t>::iterator pi = P.begin(); pi != P.end(); pi++){
          int index = pi-P.begin();
          ZZ_p Mpi = conv<ZZ_p>(M / (*pi));

          ZZ_p::init(ZZ(*pi));

          ZZ_p innerMpi = conv<ZZ_p>(M / (*pi));
          ZZ_p ZZMpiInnerInv = NTL::inv(innerMpi);
          ZZ_pEX xi = X[index] * ZZ_pEX(ZZMpiInnerInv);

          ZZ_p::init(M);

          icrti[index] = ZZ_pEX(xi) * Mpi;
  }
  end = cycles();
  // std::cout << "icrt computation: " << (end-start) << std::endl;

  // Sum all
  start = cycles();
  ZZ_pEX result;
  for(std::vector<ZZ_pEX>::iterator iter = icrti.begin(); iter != icrti.end();iter++)
    // We only add the first P.size() irst elements
    result += (*iter);
  end = cycles();
  // std::cout << "sum all: " << (end-start) << std::endl;

  // Recover original mod
  ZZ_p::init(q);
  // Set to 0 all non-used coefficients
  this->set_coeffs(NTL::deg(result)+1);//Discards all coeffs
  
  start = cycles();
  // Set new coefficients
  for(int index = 0; index <= NTL::deg(result); index++){
      ZZ r = conv<ZZ>(NTL::coeff(NTL::rep(result[index]),0));// Gambiarra
      ZZ_p value = conv<ZZ_p>(r);//This applies mod q

      this->set_coeff(index,conv<ZZ>(value));
  }
  end = cycles();
  // std::cout << "Convertion back " << (end-start) << std::endl;

  this->normalize();
  this->set_host_updated(true);
  this->set_icrt_computed(true);
  return;
}

uint64_t polynomial_get_cycles() {
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

void Polynomial::DivRem(Polynomial a,Polynomial b,Polynomial &quot,Polynomial &rem){
  // Returns x = a % b

  if(a.get_host_updated() == false && b.get_host_updated() == false){
    // Operates on GPU
    throw "DivRem for GPU not implemented yet";
  }else{
    if(!a.get_host_updated()){
        a.update_host_data();
        a.icrt();
    }

    if(!b.get_host_updated()){
        b.update_host_data();
        b.icrt();
    }

    if(check_special_rem_format(b)){
      #ifdef VERBOSE
      std::cout << "Rem in special mode."<<std::endl;
      #endif

      Polynomial lower_half;
      int half = b.deg()-1;
      for(int i = a.deg(); i > half; i--)
        quot.set_coeff(i-b.deg(),a.get_coeff(i));
      for(int i = half;i >= 0;i--)
        lower_half.set_coeff(i,a.get_coeff(i));
      lower_half.CPUSubtraction(&quot);
      rem = lower_half;
    }else{
      throw "DivRem: I don't know how to div this!";
    }
  }
}

int isPowerOfTwo (unsigned int x){
  return ((x != 0) && !(x & (x - 1)));
}

void Polynomial::BuildNthCyclotomic(Polynomial *phi,unsigned int n){

  phi->set_coeffs();

  if(isPowerOfTwo(n)){
    #ifdef VERBOSE
    std::cout << n << " is power of 2" << std::endl;
    #endif
    phi->set_coeff(0,1);
    phi->set_coeff(n,1);
    return;
  }else{
    #ifdef VERBOSE
    std::cout << n << " is not power of 2" << std::endl;
    #endif

    std::vector<Polynomial> aux_phi( n+1);

    for (cuyasheint_t i = 1; i <= n; i++) {
       Polynomial t;
       t.set_coeff(0,ZZ(1));

       for (cuyasheint_t j = 1; j <= i-1; j++)
          if (i % j == 0)
             t *= aux_phi[j];

       Polynomial mono;
       mono.set_coeff(i,ZZ(1));
       aux_phi[i] = (mono - 1)/t;

      //  cout << aux_phi[i] << "\n";
    }
    *phi = aux_phi[n];
 }
}

Polynomial Polynomial::get_phi(){
  // Returns a copy of phi
  // std::cout << "getphi!" << std::endl;
  // if(this->phi == NULL){
  //   // std::cout << "Using global phi." << std::endl;
  //   return *(this->global_phi);
  // }
  // std::cout << "Using local phi: " << this->phi->to_string() << std::endl;
  // // return *(this->phi);
  // return *(this->phi);

  return *(Polynomial::global_phi);
}

// void Polynomial::XGCD(Polynomial& d, Polynomial& s, Polynomial& t,  Polynomial& a,  Polynomial& b){
//   ZZ z;

//   if (b.is_zero()) {
//     s.set_coeffs();
//     s.set_coeff(0,1);
//     t.set_coeffs();
//     d = a;
//   }else if (a.is_zero()) {
//     t.set_coeffs();
//     t.set_coeff(0,1);
//     s.set_coeffs();
//     d = b;
//   }else {
//     long e = max(a.deg(), b.deg()) + 1;

//     Polynomial temp(e), u(e), v(e),
//           u0(e), v0(e),
//           u1(e), v1(e),
//           u2(e), v2(e), q(e);


//     (u1.set_coeff(0,1)); (v1.set_coeffs());
//     (u2.set_coeffs()); (v2.set_coeff(0,1));
//     u = a; v = b;

//     do {
//        Polynomial::DivRem(q, u, u, v);
//        std::swap(u, v);
//        u0 = u2;
//        v0 = v2;
//        temp = q * u2;
//        u2 = u1 - temp;
//        temp = q * v2;
//        v2 = v1 - temp;
//        u1 = u0;
//        v1 = v0;
//     } while (!v.is_zero());

//     d = u;
//     s = u1;
//     t = v1;
//   }

//   if (d.is_zero()) return;
//   if (d.lead_coeff() == 1) return;

//   /* make gcd monic */

//   inv(z, d.lead_coeff());
//   d = d * z;
//   s = s * z;
//   t = t * z;
// }

// void Polynomial::InvMod(Polynomial& x, const Polynomial& a, const Polynomial& f)
// {
//    if (a.deg()) >= f.deg()) || f.deg()) == 0) LogicError("InvMod: bad args");

//    Polynomial d, xx, t;

//    XGCD(d, xx, t, a, f);
//    if (!d.is_one())
//       InvModError("ZZ_pEX InvMod: can't compute multiplicative inverse");

//    x = xx;
// }
