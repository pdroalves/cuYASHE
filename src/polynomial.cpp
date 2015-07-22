#include "polynomial.h"
#include "common.h"
#include <omp.h>
#include <assert.h>

ZZ Polynomial::CRTProduct = ZZ(1);
std::vector<long> Polynomial::CRTPrimes(0);
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


    result = cudaMalloc((void**)&this->d_polyCRT,this->CRTSPACING*(this->polyCRT.size())*sizeof(long));
    #ifdef VERBOSE
    std::cout << "cudaMalloc:" << cudaGetErrorString(result) << " "<< this->CRTSPACING*(this->polyCRT.size())*sizeof(long) << " bytes" <<std::endl;
    #endif
    assert(result == cudaSuccess);


    result = cudaMemset((void*)this->d_polyCRT,0,this->CRTSPACING*(this->polyCRT.size())*sizeof(long));
    #ifdef VERBOSE
    std::cout << "cudaMemset:" << cudaGetErrorString(result) << std::endl;
    #endif
    assert(result == cudaSuccess);

    for(unsigned int i=0;i < this->polyCRT.size();i++){
        result = cudaMemcpyAsync(this->d_polyCRT+this->CRTSPACING*i, &(this->polyCRT[i][0]) , (this->polyCRT[i].size())*sizeof(long), cudaMemcpyHostToDevice,this->stream);

        #ifdef VERBOSE
        std::cout << "cudaMemcpyAsync" << i << ": " << cudaGetErrorString(result) << " "<<(this->polyCRT[i].size())*sizeof(long) << " bytes to position "<< this->CRTSPACING*i*sizeof(int) <<std::endl;
        #endif
        assert(result == cudaSuccess);
    }

    result = cudaDeviceSynchronize();
    #ifdef VERBOSE
    std::cout << cudaGetErrorString(result) << std::endl;
    #endif
    assert(result == cudaSuccess);
    this->ON_COPY = false;
    this->set_device_updated(true);
}

void Polynomial::update_host_data(){
    if(this->get_host_updated())
      return;

    cudaError_t result;

    if(this->polyCRT.size() != Polynomial::CRTPrimes.size())
      this->polyCRT.resize(Polynomial::CRTPrimes.size());

    for(unsigned int i=0;i < this->polyCRT.size();i++){
      if(this->polyCRT[i].size() != this->CRTSPACING)
        this->polyCRT[i].resize(this->CRTSPACING);

        result = cudaMemcpyAsync(&(this->polyCRT[i][0]) , this->d_polyCRT+this->CRTSPACING*i, this->CRTSPACING*sizeof(long), cudaMemcpyDeviceToHost,this->stream);
        #ifdef VERBOSE
        std::cout << "cudaMemCpy" << i << ": " << cudaGetErrorString(result) <<" "<<this->CRTSPACING*sizeof(long) << " bytes to position "<< this->CRTSPACING*i*sizeof(long) <<std::endl;
        #endif
        assert(result == cudaSuccess);
    }

    result = cudaDeviceSynchronize();
    //assert(result == cudaSuccess);

    this->set_host_updated(true);
}

void Polynomial::crt(){
    // "The product of those prime numbers should be larger than the potentially largest coefficient of polynomial c, that we will obtain as a result of a computation for accurate recovery through ICRT." produtorio_{i=1}^n (pi) > n*q^2

    // if(this->CRTProduct == NULL or this->CRTPrimes == NULL){
    //     throw -1;
    // }
    std::vector<long> P = this->CRTPrimes;
    this->polyCRT.resize(P.size());

    // Extract the coefficients to a array of ZZs
    std::vector<ZZ> array = this->get_coeffs();

    // We pick each prime
    for(std::vector<long>::iterator iter_prime = P.begin(); iter_prime != P.end(); iter_prime++){
        int P_i = iter_prime - P.begin();//Debug

        // Apply mod at each coefficient
        std::vector<long> rep = this->polyCRT[P_i];
        rep.resize(array.size());
        for(std::vector<ZZ>::iterator iter = array.begin();iter != array.end();iter++){
          int array_i = iter-array.begin();//Debug
          rep[array_i] = conv<long>(*iter % (*iter_prime));
        }

         polyCRT[P_i] = (rep);
    }

    this->set_device_updated(false);
}

void Polynomial::icrt(){
  // Escapes, if possible
  if(this->get_host_updated()){
    return;
  }else{
      this->update_host_data();
  }

  std::vector<long> P = this->CRTPrimes;
  ZZ M = this->CRTProduct;

  Polynomial icrt(this->get_mod(),this->get_phi(),this->get_crt_spacing());
  for(unsigned int i = 0; i < this->polyCRT.size();i++){
  // Convert CRT representations to Polynomial
    std::vector<ZZ> residue;
    std::copy ( (this->polyCRT[i]).begin(), (this->polyCRT[i]).begin() +7, residue.begin() );
    Polynomial xi(this->get_mod(),this->get_phi(),this->get_crt_spacing());
    xi.set_coeffs(residue);

    ZZ pi = ZZ(P[i]);
    ZZ Mpi= M/pi;
    ZZ InvMpi = NTL::InvMod(Mpi%pi,pi);

    Polynomial step = ((xi*InvMpi % pi)*Mpi) % M;

    icrt += step;
  }

  icrt %= icrt.get_phi();
  this->set_coeffs(icrt.get_coeffs());
  return;
}
void Polynomial::DivRem(Polynomial a,Polynomial b,Polynomial *quot,Polynomial *rem){
  // Returns x = a % b

  if(!a.HOST_IS_UPDATED){
      a.update_host_data();
      a.icrt();
  }

  if(!b.HOST_IS_UPDATED){
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
      quot->set_coeff(i-b.deg(),a.get_coeff(i));
      // std::cout << "(*quot): " << (*quot) << std::endl;
    for(int i = half;i >= 0;i--)
      lower_half.set_coeff(i,a.get_coeff(i));
    // std::cout << "lower_half: " << lower_half << std::endl;
    *rem = lower_half - (*quot);
  }else{
    throw "I don't know how to div this!";
  }
}
