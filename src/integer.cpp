#include "integer.h"
#include "cuda_functions.h"

void Integer::update_device_data(){
    if(this->get_device_updated())
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
    this->set_device_updated(true);
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
  std::vector<ZZ> invMpis = Polynomial::CRTInvMpi;
 
  // Discards all coefficients and prepare to receive new this->CRTSPACING coefficients

  // Iteration over all primes
  for(unsigned int i = 0; i < primes.size();i++){
    // Get a prime 
    ZZ pi = to_ZZ(primes[i]);
    ZZ Mpi = Mpis[i];
    ZZ invMpi = invMpis[i];

    this->value += Mpi*( invMpi*(this->crt_values[i]) % pi);
  }
 
 
  this->set_host_updated(true);
  this->set_icrt_computed(true);
  
  this->value %= M;
  return;
}

Polynomial Integer::operator+(Polynomial &a){
    Polynomial *p;
    p = new Polynomial();
    p->copy(a);
    CUDAFunctions::callPolynomialOPInteger( ADD,
                                            p->get_stream(),
                                            p->get_device_crt_residues(),
                                            this->get_device_crt_residues(),
                                            p->get_crt_spacing(),
                                            Polynomial::CRTPrimes.size());
    return p;
}

Polynomial Integer::operator*(Polynomial &a){

  // Apply CRT and copy data to global memory, if needed
  // #pragma omp parallel sections num_threads(2)
  {
      // #pragma omp section
      {
        #ifdef VERBOSE
          std::cout << "this: " << std::endl;
          #endif
        if(!a.get_device_updated()){
          a.update_device_data();
        }

      }
      // #pragma omp section
      {
        #ifdef VERBOSE
          std::cout << "b: " << std::endl;
          #endif
          if(!this->get_device_updated()){
          this->update_device_data();
        }
      }
  }

  cuyasheint_t *result = CUDAFunctions::callPolynomialOPInteger( MUL,
                                          a.get_stream(),
                                          a.get_device_crt_residues(),
                                          this->get_device_crt_residues(),
                                          a.get_crt_spacing(),
                                          Polynomial::CRTPrimes.size());

  Polynomial *p = new Polynomial();
  p->CRTSPACING = a.get_crt_spacing();
  p->set_device_crt_residues(result);

  p->set_host_updated(false);
  p->set_device_updated(true);
  return p;
}