#include "polynomial.h"
#include "common.h"
#include <omp.h>
#include <assert.h>
#include <string.h>

ZZ Polynomial::CRTProduct = ZZ(1);
std::vector<cuyasheint_t> Polynomial::CRTPrimes(0);
ZZ Polynomial::global_mod = ZZ(0);
Polynomial *(Polynomial::global_phi) = NULL;

void Polynomial::update_device_data(){

    if(this->get_device_updated())
      return;

    #ifdef VERBOSE
    std::cout << "Copying data to GPU." << std::endl;
    #endif

    cudaError_t result;
    this->ON_COPY = true;


    result = cudaMalloc((void**)&this->d_polyCRT,this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t));
    #ifdef VERBOSE
    std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t) << " bytes" <<std::endl;
    #endif
    assert(result == cudaSuccess);


    // result = cudaMemset((void*)this->d_polyCRT,0,this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t));
    // #ifdef VERBOSE
    // std::cout << "cudaMemset:" << cudaGetErrorString(result) << std::endl;
    // #endif
    // assert(result == cudaSuccess);

    cuyasheint_t *aux;
    // aux = (cuyasheint_t*)malloc(this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t));
    aux = (cuyasheint_t*)calloc(this->CRTSPACING*(this->polyCRT.size()),sizeof(cuyasheint_t));
    for(unsigned int i=0;i < this->polyCRT.size();i++){
      memcpy(aux+this->CRTSPACING*i,&(this->polyCRT[i][0]),(this->polyCRT[i].size())*sizeof(cuyasheint_t));
    }

    result = cudaMemcpy(this->d_polyCRT, aux , this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t), cudaMemcpyHostToDevice);

    #ifdef VERBOSE
    std::cout << "cudaMemcpyAsync" << i << ": " << cudaGetErrorString(result) << " "<<(this->polyCRT[i].size())*sizeof(cuyasheint_t) << " bytes to position "<< this->CRTSPACING*i*sizeof(int) <<std::endl;
    #endif
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

    cudaError_t result;

    if(this->polyCRT.size() != Polynomial::CRTPrimes.size())
      this->polyCRT.resize(Polynomial::CRTPrimes.size());

    // Copy all data to host
    cuyasheint_t *aux;
    // aux = (cuyasheint_t*) calloc (this->polyCRT.size()*this->CRTSPACING,sizeof(cuyasheint_t));
    aux = (cuyasheint_t*) malloc (this->polyCRT.size()*this->CRTSPACING*sizeof(cuyasheint_t));
    result = cudaMemcpy(aux , this->d_polyCRT, this->polyCRT.size()*this->CRTSPACING*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
    #ifdef VERBOSE
    std::cout << "cudaMemCpy: " << cudaGetErrorString(result) <<" "<< this->polyCRT.size()*this->CRTSPACING*sizeof(cuyasheint_t) <<std::endl;
    #endif
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
}
uint64_t get_cycles(void) {
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
void Polynomial::icrt(){
  // Escapes, if possible
  if(this->get_host_updated())
    return;
  else
    this->update_host_data();

  std::vector<cuyasheint_t> P = this->CRTPrimes;
  ZZ M = this->CRTProduct;

  // Polynomial icrt(this->get_mod(),this->get_phi(),this->get_crt_spacing());
  this->set_coeffs();//Discards all coeffs

  // 4M cycles per iteration
  for(unsigned int i = 0; i < this->CRTPrimes.size();i++){
    // uint64_t start_cycle = get_cycles();
    // Convert CRT representations to Polynomial
    // Polynomial xi(this->get_mod(),this->get_phi(),this->get_crt_spacing());
  
    Polynomial xi(this->get_mod(),this->get_phi(),this->get_crt_spacing());
    xi.set_coeffs(this->polyCRT[i]);
    // Asserts that each residue is in the correct field
    ZZ pi = ZZ(P[i]);
    xi %= pi;

    ZZ Mpi= M/pi;
    //
    ZZ InvMpi = NTL::InvMod(Mpi%pi,pi);
    //



    xi = ((xi*InvMpi)%pi)*Mpi;

    this->CPUAddition(&xi);
    // uint64_t end_cycle = get_cycles();
    // std::cout << "Cycles for each icrt iteration: " << (end_cycle-start_cycle) << std::endl;
  }

  (*this) %= M;
  this->normalize();
  this->set_host_updated(true);
  return;
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

  for(int i =0; i <= phi->deg(); i++)
    phi->set_coeff(i,0);
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
