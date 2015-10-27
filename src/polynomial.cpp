#include <omp.h>
#include <assert.h>
#include <string.h>
#include "polynomial.h"
#include "integer.h"
#include "settings.h"
#include "common.h"

ZZ Polynomial::CRTProduct = ZZ(1);
std::vector<cuyasheint_t> Polynomial::CRTPrimes(0);
std::vector<ZZ> Polynomial::CRTMpi;
std::vector<ZZ> Polynomial::CRTInvMpi;
ZZ Polynomial::global_mod = ZZ(0);
Polynomial *(Polynomial::global_phi) = NULL;
bool Polynomial::phi_set = false;



Polynomial Polynomial::operator+(Polynomial &b){
  return common_addition<Polynomial>(this,&b);
}

Polynomial Polynomial::operator+=(Polynomial &b){
  common_addition_inplace<Polynomial>(this,&b);
  return *this;
}

Polynomial Polynomial::operator*(Polynomial &b){
  return common_multiplication<Polynomial>(this,&b);
}

Polynomial Polynomial::operator*(ZZ b){
      Polynomial p(*this);
      if(!p.get_host_updated()){
        // Operate on device
        Integer I = b;
        CUDAFunctions::callPolynomialOPInteger(MUL,
                                                p.get_stream(),
                                                p.get_device_crt_residues(),
                                                I.get_device_crt_residues(),
                                                p.CRTSPACING,Polynomial::CRTPrimes.size());
      }else{
        //#pragma omp parallel for
        for(int i = 0; i <= p.deg();i++)
          p.set_coeff(i,p.get_coeff(i)*b);
        p.set_device_updated(false);
      }
      return p;
    }

void Polynomial::operator delete(void *ptr){
      Polynomial *p = (Polynomial*)ptr;

      if(p->get_device_crt_residues() != 0x0){
        std::cout << "cudaFree!" << std::endl;
        cudaError_t result = cudaFree(p->get_device_crt_residues());
        assert(result == cudaSuccess);
      }

      // free(ptr);
    }

void Polynomial::copy_device_crt_residues(Polynomial &b){
      uint64_t start,end;

      start = get_cycles();
      if(!b.get_device_updated())
        return;

      // std::cout << "Will copy residues on device memory" << std::endl;
      // Discards the existing values
      cudaError_t result;
      if(d_polyCRT != 0x0){
        result = cudaFree(d_polyCRT);
        assert(result == cudaSuccess);
      }

      // Adjusts CRTSPACING
      // Here we don't use update_crt_spacing(). The reason
      // is: speed and context.
      // update_crt_spacing() won't just update the spacing, but
      // also update device data. And in this context, we know
      // that there is no relevant data to update.
      this->CRTSPACING = b.get_crt_spacing();

      result = cudaMalloc((void**)&d_polyCRT,
                          this->get_crt_spacing()*(Polynomial::CRTPrimes.size())*sizeof(cuyasheint_t));
      assert(result == cudaSuccess);
      result = cudaMemcpy(d_polyCRT,
                          b.get_device_crt_residues(),
                          this->get_crt_spacing()*(Polynomial::CRTPrimes.size())*sizeof(cuyasheint_t),
                          cudaMemcpyDeviceToDevice);      
      assert(result == cudaSuccess);
      end = get_cycles();
      std::cout << (end-start) << " cycles" << std::endl;
    }

void Polynomial::update_device_data(unsigned int usable_ratio){
 

  // "usable_ratio" could be used to avoid unnecessary copies
  if(this->get_device_updated())
    return;
  else if(!this->get_crt_computed()){
    this->crt();
  }

   #ifdef VERBOSEMEMORYCOPY
  uint64_t start = get_cycles();
  #endif

  #if defined(VERBOSE) || defined(VERBOSEMEMORYCOPY) 
  std::cout << "Copying data to GPU." << std::endl;
  #endif


  this->ON_COPY = true;
  // Updated CRTSPACING    
  // Verifica se o espaçamento é válido. Se não for, ajusta.
  if(this->get_crt_spacing() < std::max(this->deg()+1,(Polynomial::global_phi->deg()+1))){
    const int new_spacing = std::max(this->deg()+1,(Polynomial::global_phi->deg()+1));

    cuyasheint_t *d_pointer;
    cudaError_t result = cudaMalloc((void**)&d_pointer,new_spacing*(CRTPrimes.size())*sizeof(cuyasheint_t));        
    assert(result == cudaSuccess);

    this->set_device_crt_residues(d_pointer);
    this->set_crt_spacing(new_spacing);
  }

  // cuyasheint_t *d_data;
  // cudaError_t result = cudaMalloc((void**)&d_data,this->get_crt_spacing()*(this->polyCRT.size())*sizeof(cuyasheint_t));
  // assert(result == cudaSuccess);

  // cuyasheint_t *aux;
  // // aux = (cuyasheint_t*)malloc(this->CRTSPACING*(this->polyCRT.size())*sizeof(cuyasheint_t));
  // aux = (cuyasheint_t*)calloc(this->get_crt_spacing()*(1+this->polyCRT.size()),sizeof(cuyasheint_t));
  // for(unsigned int i=0;i < this->polyCRT.size();i++)
  //   // memcpy(aux+this->get_crt_spacing()*i,&(this->polyCRT[i][0]),(this->polyCRT[i].size()/usable_ratio)*sizeof(cuyasheint_t));
  //   memcpy(aux+this->get_crt_spacing()*i,&(this->polyCRT[i][0]),(this->polyCRT[i].size())*sizeof(cuyasheint_t));
  

  // cudaError_t result = cudaMemcpy(this->get_device_crt_residues(), aux , this->get_crt_spacing()*(this->polyCRT.size())*sizeof(cuyasheint_t), cudaMemcpyHostToDevice);
  // assert(result == cudaSuccess);

  cudaError_t result = cudaMemsetAsync( this->get_device_crt_residues(),
                                        0,
                                        this->get_crt_spacing()*(this->polyCRT.size())*sizeof(cuyasheint_t),
                                        this->get_stream());
  assert(result == cudaSuccess);  
  for(unsigned int i=0;i < this->polyCRT.size();i++){
    cudaError_t result = cudaMemcpyAsync( this->get_device_crt_residues()+this->get_crt_spacing()*i,
                                          &(this->polyCRT[i][0]) ,
                                          (this->polyCRT[i].size())*sizeof(cuyasheint_t),
                                          cudaMemcpyHostToDevice,
                                          this->get_stream());
    assert(result == cudaSuccess);
  }


  // #warning "free(aux) commented"
  // free(aux);
  // cudaError_t result = cudaDeviceSynchronize();
  // assert(result == cudaSuccess);
  this->ON_COPY = false;
  this->set_device_updated(true);

  #ifdef VERBOSEMEMORYCOPY
  uint64_t end = get_cycles();
  std::cout << "Cycles needed: " << (end-start) <<std::endl;
  #endif
}

void Polynomial::update_host_data(){
  
    if(this->get_host_updated())
      return;
    
    #ifdef VERBOSEMEMORYCOPY
    std::cout << "Copying data to the CPU..."<<std::endl;
    uint64_t start = get_cycles();
    #endif

    #ifdef VERBOSE
    std::cout << "Copying data to CPU." << std::endl;
    #endif

    cudaError_t result;

    assert(Polynomial::CRTPrimes.size() > 0);
    if(this->polyCRT.size() != Polynomial::CRTPrimes.size())
      this->polyCRT.resize(Polynomial::CRTPrimes.size());

    for(unsigned int i=0;i < Polynomial::CRTPrimes.size();i++){
      if(this->polyCRT[i].size() < (unsigned int)(this->get_crt_spacing)())
        this->polyCRT[i].resize(this->get_crt_spacing());
      result = cudaMemcpyAsync(&this->polyCRT[i][0] ,
                                this->d_polyCRT + i*this->get_crt_spacing(),
                                this->get_crt_spacing()*sizeof(cuyasheint_t),
                                cudaMemcpyDeviceToHost,
                                this->get_stream());
      if(result != cudaSuccess)
        std::cout << this->get_crt_spacing() << std::endl;
      assert(result == cudaSuccess);
    }
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
    this->set_host_updated(true);
    this->icrt();
    
    #ifdef VERBOSEMEMORYCOPY
    uint64_t end = get_cycles();
    std::cout << "Cycles needed: " << (end-start) <<std::endl;
    #endif
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
    
    // Set the quantity of expected residues
    std::vector<cuyasheint_t> P = this->CRTPrimes;
    this->polyCRT.resize(P.size());

    // Updated CRTSPACINg
    // if(this->get_crt_spacing() < (Polynomial::global_phi->deg()))
    //   this->update_crt_spacing((Polynomial::global_phi->deg()));


    // Extract the coefficients to a array of ZZs
    std::vector<ZZ> array = this->get_coeffs();

    // We pick each prime
    #pragma omp parallel for  
    for(unsigned int i = 0; i < P.size();i++){
      this->polyCRT[i].resize(array.size());

      for(unsigned int j = 0; j < array.size();j++)
        this->polyCRT[i][j] = conv<cuyasheint_t>(array[j] % P[i]);
      
    }

    // for(unsigned int j = 0; j < polyCRT.size();j++){
    //   std::cout << "CRT Polynomial residue "<< j << ":" << std::endl; 
    //   for(unsigned int i = 0; i < polyCRT[j].size() ;i++)
    //     std::cout << polyCRT[j][i] << " ";
    //   std::cout << std::endl << std::endl;
    // }

    this->set_host_updated(true);
    this->set_device_updated(false);
    this->set_crt_computed(true);
}

void Polynomial::icrt(){

  #ifdef CYCLECOUNTING
  uint64_t start,end;
  uint64_t start_total,end_total;
  start_total = get_cycles();
  
  start = get_cycles();
  #endif
  // Escapes, if possible
  if(this->get_icrt_computed())
    return;
  else if(!this->get_host_updated()){
    this->set_icrt_computed(true);//If we do not set this, we get a infinite loop
    this->update_host_data();
  }
  #ifdef CYCLECOUNTING
  end = get_cycles();
  std::cout << "Cycles for host update: " << (end-start) << std::endl;
  #endif
    // for(unsigned int j = 0; j < polyCRT.size();j++){
    //   std::cout << "ICRT Polynomial residue"<< j << ":" << std::endl; 
    //   for(unsigned int i = 0; i < polyCRT[j].size() ;i++)
    //     std::cout << polyCRT[j][i] << " ";
    //   std::cout << std::endl << std::endl;
    // }
  #ifdef CYCLECOUNTING
  start = get_cycles();
  #endif
  
  ZZ M = Polynomial::CRTProduct;
  std::vector<cuyasheint_t> primes = Polynomial::CRTPrimes;
  std::vector<ZZ> Mpis = Polynomial::CRTMpi;
  std::vector<ZZ> invMpis = Polynomial::CRTInvMpi;
 
  #ifdef CYCLECOUNTING
  end = get_cycles();
  std::cout << "Cycles to load CRT variables: " << (end-start) << std::endl;
  std::cout << "#Primes: " << primes.size() << std::endl;
  #endif

  // Discards all coefficients and prepare to receive new this->CRTSPACING coefficients

  this->set_coeffs(this->CRTSPACING);
  
  #ifdef CYCLECOUNTING
  uint64_t start_iteration,end_iteration;
  start = get_cycles();
  #endif
  // Iteration over all primes
  for(unsigned int i = 0; i < primes.size();i++){
    // Get a prime
    #ifdef CYCLECOUNTING
    start_iteration = get_cycles();
    #endif
  
    ZZ pi = to_ZZ(primes[i]);
    ZZ Mpi = Mpis[i];
    ZZ invMpi = invMpis[i];
  
    #ifdef CYCLECOUNTING
    end_iteration = get_cycles();
    std::cout << "Cycles to load iteration variables: " << (end_iteration-start_iteration) << std::endl;
   
    start_iteration = get_cycles();
    #endif

    if(&this->polyCRT[i] == NULL)
      std::cout << &this->polyCRT[i] << std::endl;

    // Iteration over coefficients
    #pragma omp parallel for
    for(unsigned int j = 0; j < this->polyCRT[i].size();j++)
      this->coefs[j] += Mpi*( invMpi*(this->polyCRT[i][j]) % pi);  
    #ifdef CYCLECOUNTING
    end_iteration = get_cycles();
    std::cout << "Cycles in the inner loop: " << (end_iteration-start_iteration) << std::endl;
    #endif
    
    // std::cout << "this: " << std::endl; 
    // for(unsigned int i = 0; i < this->coefs.size() ;i++)
    //   std::cout << this->coefs[i] << " ";
    // std::cout << std::endl << std::endl;
  }
  #ifdef CYCLECOUNTING
  end = get_cycles();
  std::cout << "Cycles in iterations: " << (end-start) << std::endl;
 
  start = get_cycles();
  #endif
 
  *this %= M;
 
  #ifdef CYCLECOUNTING
  end = get_cycles();
  std::cout << "Cycles to compute mod M: " << (end-start) << std::endl;
  #endif


  this->normalize();

  // Correction on negative values
  for(unsigned int i = 0; i < this->coefs.size(); i++){
    // std::cout << this->coefs[i] << " > " << Polynomial::global_mod << "?" << std::endl;
    this->coefs[i] = (this->coefs[i] > Polynomial::global_mod)? (this->coefs[i]-M) % Polynomial::global_mod : this->coefs[i]; 
    // std::cout << this->coefs[i] << std::endl;
  }
  // this->update_crt_spacing(this->deg()+1);
  
  this->set_host_updated(true);
  this->set_icrt_computed(true);
  // Updated CRTSPACINg
  if(this->get_crt_spacing() < (Polynomial::global_phi->deg()))
    this->update_crt_spacing((Polynomial::global_phi->deg()));

  #ifdef CYCLECOUNTING
  end_total = get_cycles();
  std::cout << "Cycles for icrt: " << (end_total-start_total) << std::endl<< std::endl;
  #endif
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
    // No need to reduce
    if(a.deg() <= 0)
      return;
    if(check_special_rem_format(b)){
      #ifdef VERBOSE
      std::cout << "Rem in special mode."<<std::endl;
      #endif

      const unsigned int half = b.deg()-1;     

      for(unsigned int i = 0;i <= half;i++)
        rem.set_coeff(i,a.get_coeff(i)-a.get_coeff(i+half+1));
      

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