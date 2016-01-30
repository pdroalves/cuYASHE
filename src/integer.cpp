#include "polynomial.h"
#include "integer.h"
#include "cuda_functions.h"

void Integer::update_device_data(){
    if(this->get_crt_computed())
      return;
    else if(!this->get_crt_computed())
      this->crt();

    #ifdef VERBOSE
    std::cout << "Copying data to GPU." << std::endl;
    #endif

    cudaError_t result;
    this->ON_COPY = true;

    result = cudaMalloc((void**)&this->d_crt_values,
                          (this->crt_values.size())*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

    result = cudaMemcpyAsync(this->d_crt_values,
                            &crt_values[0] ,
                            (this->crt_values.size())*sizeof(cuyasheint_t),
                            cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    // result = cudaDeviceSynchronize();
    // assert(result == cudaSuccess);

    this->ON_COPY = false;
    this->set_crt_residues_computed(true);
}

void Integer::update_host_data(){

    if(this->get_host_updated())
      return;

    #ifdef VERBOSE
    std::cout << "Copying data to CPU." << std::endl;
    #endif

    cudaError_t result;

    assert(Polynomial::CRTPrimes.size() > 0);
    if(this->crt_values.size() != Polynomial::CRTPrimes.size())
      this->crt_values.resize(Polynomial::CRTPrimes.size());

    result = cudaMemcpy(&this->crt_values[0],
                        this->d_crt_values,
                        Polynomial::CRTPrimes.size()*sizeof(cuyasheint_t), 
                        cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);

    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
    this->set_host_updated(true);
    this->icrt();
}

void Integer::crt(){
      // Escapes, if possible

      if(this->get_crt_computed())
        return;
      else if(!this->get_host_updated())
        throw "host is not updated";

      // Set the quantity of expected residues
      std::vector<cuyasheint_t> P = Polynomial::CRTPrimes;
      this->crt_values.resize(P.size());

      // We pick each prime
      // #pragma omp parallel for  
      for(unsigned int i = 0; i < P.size();i++)
          crt_values[i] = conv<cuyasheint_t>(value % P[i]);
          

      this->set_crt_residues_computed(false);
      this->set_crt_computed(true);
    }

void Integer::icrt(){

  // Escapes, if possible
  if(this->get_icrt_computed())
    return;
  else if(this->get_host_updated()){
    this->set_icrt_computed(true);
    return;
  } 
  
  ZZ M = Polynomial::CRTProduct;
  std::vector<cuyasheint_t> primes = Polynomial::CRTPrimes;
  std::vector<ZZ> Mpis = Polynomial::CRTMpi;
  std::vector<cuyasheint_t> invMpis = Polynomial::CRTInvMpi;
 
  // Discards all coefficients and prepare to receive new this->CRTSPACING coefficients

  // Iteration over all primes
  for(unsigned int i = 0; i < primes.size();i++){
    // Get a prime 
    cuyasheint_t pi = primes[i];
    ZZ Mpi = Mpis[i];
    cuyasheint_t invMpi = invMpis[i];
    
    this->value += Mpi*( invMpi*(this->crt_values[i]) % pi);
  }
 
 
  this->set_host_updated(true);
  this->set_icrt_computed(true);
  
  this->value %= M;
  return;
}

Polynomial Integer::operator+(Polynomial &a){
  Polynomial p = Polynomial(a);
  if(a.get_icrt_computed()){
    assert(p.d_bn_coefs);
    CUDAFunctions::callPolynomialOPDigit( ADD,
                                        p.get_stream(),
                                        p.d_bn_coefs,
                                        this->digits,
                                        p.deg()+1);
    p.set_crt_computed(false);
    p.set_host_updated(false);
  }
  else{
    CUDAFunctions::callPolynomialOPInteger( ADD,
                                            p.get_stream(),
                                            p.get_device_crt_residues(),
                                            this->get_device_crt_residues(),
                                            p.get_crt_spacing(),
                                            Polynomial::CRTPrimes.size());   
  }
  return p;
}

Polynomial Integer::operator-(Polynomial &a){
  Polynomial p = Polynomial(a);
  if(a.get_icrt_computed()){
    assert(p.get_crt_computed());
    assert(p.d_bn_coefs);
    CUDAFunctions::callPolynomialOPDigit( SUB,
                                        p.get_stream(),
                                        p.d_bn_coefs,
                                        this->digits,
                                        p.deg()+1);
    p.set_crt_computed(false);
    p.set_host_updated(false);
  }
  else{
    CUDAFunctions::callPolynomialOPInteger( SUB,
                                            p.get_stream(),
                                            p.get_device_crt_residues(),
                                            this->get_device_crt_residues(),
                                            p.get_crt_spacing(),
                                            Polynomial::CRTPrimes.size());   
  }
  return p;
}

Polynomial Integer::operator*(Polynomial &a){
  Polynomial p;
  p.copy(a);

  // Apply CRT and copy data to global memory, if needed
  if(get_crt_computed()){
    // #pragma omp parallel sections num_threads(2)
    {
        // #pragma omp section
        {
          #ifdef VERBOSE
            std::cout << "this: " << std::endl;
            #endif
          if(!a.get_crt_computed()){
            a.update_device_data();
          }

        }
        // #pragma omp section
        {
          #ifdef VERBOSE
            std::cout << "b: " << std::endl;
            #endif
            if(!this->get_crt_computed()){
            this->update_device_data();
          }
        }
    }

    cuyasheint_t *d_result = CUDAFunctions::callPolynomialOPInteger( MUL,
                                                                    a.get_stream(),
                                                                    a.get_device_crt_residues(),
                                                                    this->get_device_crt_residues(),
                                                                    a.get_crt_spacing(),
                                                                    Polynomial::CRTPrimes.size()
                                                                  );
    cudaError_t result = cudaFree(p.get_device_crt_residues());
    assert(result == cudaSuccess);

    p.set_device_crt_residues(d_result);
    p.set_host_updated(false);
    p.set_icrt_computed(false);
    p.set_crt_computed(true);
      
  }else if (get_icrt_computed()){
    assert(p.d_bn_coefs);
    CUDAFunctions::callPolynomialOPDigit( MUL,
                                        p.get_stream(),
                                        p.d_bn_coefs,
                                        this->digits,
                                        p.deg()+1
                                      );
    p.set_crt_computed(false);
    p.set_host_updated(false);
  }else{
    throw "Don't know how to multiply";
  }
  return p;
}