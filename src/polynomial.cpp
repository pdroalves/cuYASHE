#include "polynomial.h"
#include "settings.h"
#include <omp.h>
#include <assert.h>
#include <string.h>

ZZ Polynomial::CRTProduct = ZZ(1);
std::vector<cuyasheint_t> Polynomial::CRTPrimes(0);
ZZ Polynomial::global_mod = ZZ(0);
Polynomial *(Polynomial::global_phi) = NULL;
bool Polynomial::phi_set = false;



Polynomial Polynomial::operator+(Polynomial b){
  return common_addition<Polynomial>(this,&b);
}

Polynomial Polynomial::operator+=(Polynomial b){
  common_addition_inplace<Polynomial>(this,&b);
  return *this;
}

Polynomial Polynomial::operator*(Polynomial b){
  return common_multiplication<Polynomial>(this,&b);
}
void Polynomial::update_device_data(unsigned int usable_ratio){
  // "usable_ratio" should be used to avoid unnecessary copies
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
    aux = (cuyasheint_t*)calloc(this->CRTSPACING*(1+this->polyCRT.size()),sizeof(cuyasheint_t));
    for(unsigned int i=0;i < this->polyCRT.size();i++){
      // memcpy(aux+this->CRTSPACING*i,&(this->polyCRT[i][0]),(this->polyCRT[i].size()/usable_ratio)*sizeof(cuyasheint_t));
      memcpy(aux+this->CRTSPACING*i,&(this->polyCRT[i][0]),(this->polyCRT[i].size())*sizeof(cuyasheint_t));
    }

    result = cudaMemcpy(this->d_polyCRT, aux , this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
    // #warning "free(aux) commented"
    // free(aux);
    this->ON_COPY = false;
    this->set_device_updated(true);
    assert(this->get_device_updated() == true);
}

void Polynomial::update_host_data(){
    // uint64_t start = cycles();

    if(this->get_host_updated())
      return;
    // else if(!this->get_icrt_computed())
      // this->icrt();

    #ifdef VERBOSE
    std::cout << "Copying data to CPU." << std::endl;
    #endif

    cudaError_t result;

    assert(Polynomial::CRTPrimes.size() > 0);
    if(this->polyCRT.size() != Polynomial::CRTPrimes.size())
      this->polyCRT.resize(Polynomial::CRTPrimes.size());

    // Copy all data to host
    // cuyasheint_t *aux;
    // // aux = (cuyasheint_t*) calloc (this->polyCRT.size()*this->CRTSPACING,sizeof(cuyasheint_t));
    // aux = (cuyasheint_t*) malloc (this->polyCRT.size()*this->CRTSPACING*sizeof(cuyasheint_t));
    // result = cudaMemcpy(aux , this->d_polyCRT, this->polyCRT.size()*this->CRTSPACING*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
    // assert(result == cudaSuccess);
    // // result = cudaDeviceSynchronize();
    // // assert(result == cudaSuccess);
    // //
    // for(unsigned int i=0;i < this->polyCRT.size();i++){
    //   // if(this->polyCRT[i].size() != (unsigned int)(this->CRTSPACING))
    //     // this->polyCRT[i].resize(this->CRTSPACING);
    //     // for(unsigned int j=0; j < (unsigned int)(this->CRTSPACING);j++){
    //       // this->polyCRT[i][j] = aux[j+i*this->CRTSPACING];
    //     // }

    //   this->polyCRT[i][0] = aux[i*this->CRTSPACING];
    // }

    for(unsigned int i=0;i < this->polyCRT.size();i++){
      if(this->polyCRT[i].size() != (unsigned int)(this->CRTSPACING))
        this->polyCRT[i].resize(this->CRTSPACING);
      result = cudaMemcpy(&this->polyCRT[i][0] , this->d_polyCRT + i*this->CRTSPACING, this->CRTSPACING*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
      if(result != cudaSuccess)
        std::cout << "Catch!" <<std::endl;
      assert(result == cudaSuccess);
    }
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
    this->set_host_updated(true);
  // uint64_t end = cycles();
  // std::cout << "update host " << (end-start) << std::endl;
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
        // std::cout << "Original: " << *iter << " % " << *iter_prime << " > rep : " << rep[array_i] << ", " << std::endl;;
      }

       polyCRT[index] = (rep);
    }

    // std::cout << "Polynomial 0: " << std::endl; 
    // for(unsigned int i = 0; i < polyCRT[0].size() ;i++)
      // std::cout << polyCRT[0][i] << " ";
    // std::cout << std::endl << std::endl;

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

  ZZ M = Polynomial::CRTProduct;
  std::vector<cuyasheint_t> primes = Polynomial::CRTPrimes;

  // Discards all coefficients and prepare to receive new this->CRTSPACING coefficients
  this->set_coeffs(this->CRTSPACING);

  // Iteration over all primes
  for(unsigned int i = 0; i < primes.size();i++){
    // Get a prime
    ZZ pi = ZZ(primes[i]);

    ZZ Mpi = M/pi;
    ZZ invMpi = NTL::InvMod(Mpi%pi,pi);

    // Iteration over coefficients
    for(unsigned int j = 0; j < this->polyCRT[i].size();j++){
      // std::cout << this->polyCRT[i][j] << " ";
      this->coefs[j] += Mpi*( invMpi*(this->polyCRT[i][j]) % pi);  
    }
    // std::cout << std::endl;
    
  }

  *this %= M;

  this->normalize();
  this->set_host_updated(true);
  this->set_icrt_computed(true);
  return;
}

// void Polynomial::icrt(){
//   // Escapes, if possible
//   if(this->get_host_updated())
//     return;
//   else
//     this->update_host_data();

//   std::cout << "Polynomial 0: " << std::endl; 
//   for(unsigned int i = 0; i < polyCRT[0].size();i++)
//     std::cout << polyCRT[0][i] << " ";
//   std::cout << std::endl << std::endl;

//   std::vector<cuyasheint_t> P = this->CRTPrimes;
//   ZZ M = this->CRTProduct;

//   // Polynomial icrt(this->get_mod(),this->get_phi(),this->get_crt_spacing());
//   this->set_coeffs();//Discards all coeffs

//   // 4M cycles per iteration
//   for(unsigned int i = 0; i < this->CRTPrimes.size();i++){
//     // uint64_t start_cycle = get_cycles();
//     // Convert CRT representations to Polynomial
//     // Polynomial xi(this->get_mod(),this->get_phi(),this->get_crt_spacing());
  
//     Polynomial xi(this->get_mod(),this->get_phi(),this->get_crt_spacing());
//     xi.set_coeffs(this->polyCRT[i]);
//     // Asserts that each residue is in the correct field
//     ZZ pi = ZZ(P[i]);
//     xi %= pi;

//     ZZ Mpi= M/pi;
//     //
//     ZZ InvMpi = NTL::InvMod(Mpi%pi,pi);
//     //



//     xi = ((xi*InvMpi)%pi)*Mpi;

//     this->CPUAddition(&xi);
//     // uint64_t end_cycle = get_cycles();
//     // std::cout << "Cycles for each icrt iteration: " << (end_cycle-start_cycle) << std::endl;
//   }

//   (*this) %= M;
//   this->normalize();
//   this->set_host_updated(true);
//   return;
// }

// uint64_t polynomial_get_cycles() {
//   unsigned int hi, lo;
//   asm (
//     "cpuid\n\t"/*serialize*/
//     "rdtsc\n\t"/*read the clock*/
//     "mov %%edx, %0\n\t"
//     "mov %%eax, %1\n\t" 
//     : "=r" (hi), "=r" (lo):: "%rax", "%rbx", "%rcx", "%rdx"
//   );
//   return ((uint64_t) lo) | (((uint64_t) hi) << 32);
// }

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
      return *(Polynomial::global_phi);
    }