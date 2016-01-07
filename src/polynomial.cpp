#include <omp.h>
#include <assert.h>
#include <string.h>
#include "polynomial.h"
#include "integer.h"
#include "settings.h"
#include "common.h"
#include "cuda_bn.h"

ZZ Polynomial::CRTProduct = ZZ(1);
std::vector<cuyasheint_t> Polynomial::CRTPrimes(0);
std::vector<ZZ> Polynomial::CRTMpi;
std::vector<cuyasheint_t> Polynomial::CRTInvMpi;
ZZ Polynomial::global_mod = ZZ(0);
Polynomial *(Polynomial::global_phi) = NULL;
bool Polynomial::phi_set = false;
std::map<ZZ, bn_t*> Polynomial::reciprocals;

const uint32_t PRIMES_BUCKET[] = {536870909, 536870879, 536870869, 536870849, 536870839, 536870837, 536870819, 536870813, 536870791, 536870779, 536870767, 536870743, 536870729, 536870723, 536870717, 536870701, 536870683, 536870657, 536870641, 536870627, 536870611, 536870603, 536870599, 536870573, 536870569, 536870563, 536870561, 536870513, 536870501, 536870497, 536870473, 536870401, 536870363, 536870317, 536870303, 536870297, 536870273, 536870267, 536870239, 536870233, 536870219, 536870171, 536870167, 536870153, 536870123, 536870063, 536870057, 536870041, 536870027, 536869999, 536869951, 536869943, 536869937, 536869919, 536869901, 536869891, 536869831, 536869829, 536869793, 536869787, 536869777, 536869771, 536869769, 536869747, 536869693, 536869679, 536869651, 536869637, 536869633, 536869631, 536869607, 536869603, 536869589, 536869583, 536869573, 536869559, 536869549, 536869523, 536869483, 536869471, 536869447, 536869423, 536869409, 536869387, 536869331, 536869283, 536869247, 536869217, 536869189, 536869159, 536869153, 536869117, 536869097, 536869043, 536868979, 536868977, 536868973, 536868961, 536868953, 536868901};
const int PRIMES_BUCKET_SIZE = 201;

Polynomial Polynomial::operator+(Polynomial &b){
  Polynomial p = common_addition<Polynomial>(this,&b);

  ZZ M = Polynomial::CRTProduct;  
  p.icrt();
  // p.modn(M);
  p.crt();
  return p;
}

Polynomial Polynomial::operator+=(Polynomial &b){
  common_addition_inplace<Polynomial>(this,&b);

  ZZ M = Polynomial::CRTProduct;  
  this->icrt();
  // this->modn(M);
  this->crt();
  return *this;
}

Polynomial Polynomial::operator*(Polynomial &b){
  Polynomial p = common_multiplication<Polynomial>(this,&b);
 
  ZZ M = Polynomial::CRTProduct;  
  p.icrt();
  // p.modn(M);
  p.crt();
  return p;
}

Polynomial Polynomial::operator*(ZZ b){
      Polynomial p(*this);
      if(!p.get_host_updated()){
        // Operate on device
        Integer I = b;
        return I*p;
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
    try
    {
      // std::cout << "Delete: cudaFree" << std::endl;
      cudaError_t result = cudaFree(p->get_device_crt_residues());
      if(result != cudaSuccess)
        throw string( cudaGetErrorString(result));
      
    }catch(string s){
      #ifdef VERBOSE
      std::cout << "Exception at cudaFree: " << s << std::endl;
      #endif 
      cudaGetLastError();//Reset last error
    }
  }


  for(unsigned int i = 0; i < p->bn_coefs.size();i++){
    try
    {
      // bn_free(p->bn_coefs[i]);
    }catch(string s){
      #ifdef VERBOSE
      std::cout << "Exception at cudaFree: " << s << std::endl;
      #endif 
      cudaGetLastError();//Reset last error
    }
  }
  p->bn_coefs.clear();
  // free(ptr);
}

void Polynomial::copy_device_crt_residues(Polynomial &b){
  // uint64_t start,end;

  // start = get_cycles();
  if(!b.get_device_updated())
    return;

  // std::cout << "Will copy residues on device memory" << std::endl;


  // Adjusts CRTSPACING
  // Here we don't use update_crt_spacing(). The reason
  // is: speed and context.
  // update_crt_spacing() may not just update the spacing, but
  // also update device data. And in this context, we know
  // that there is no relevant data to update.
  cuyasheint_t *aux;
  this->CRTSPACING = b.get_crt_spacing();

  cudaError_t result = cudaMalloc((void**)&aux,
                      this->get_crt_spacing()*(Polynomial::CRTPrimes.size())*sizeof(cuyasheint_t));
  assert(result == cudaSuccess);
  result = cudaMemcpyAsync(aux,
                      b.get_device_crt_residues(),
                      this->get_crt_spacing()*(Polynomial::CRTPrimes.size())*sizeof(cuyasheint_t),
                      cudaMemcpyDeviceToDevice);      
  assert(result == cudaSuccess);

  this->set_device_crt_residues(aux);
  // end = get_cycles();
  // std::cout << (end-start) << " cycles" << std::endl;
}
/**
 * Initiates a new bn_t object
 * @param a input: operand
 */

////////////////////////
// Auxiliar functions //
////////////////////////
/**
 * get_words converts a NTL big integer
 * in our bn_t format
 * @param b output: word representation
 * @param a input: operand
 */
__host__ void get_words(bn_t *b,ZZ a){
  cudaError_t result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);

  bn_new(b);
  for(ZZ x = a; x > 0; x=(x>>WORD),b->used++){
    if(b->used >= b->alloc)
      bn_grow(b,b->alloc+STD_BNT_ALLOC);
    b->dp[b->used] = conv<uint32_t>(x&UINT32_MAX);
  }
}

/**
 * Convert an array of words into a NTL ZZ
 * @param  a input:array of words
 * @return   output: NTL ZZ
 */
__host__ ZZ get_ZZ(bn_t *a){
  ZZ b = conv<ZZ>(0);
  for(int i = a->used-1; i >= 0;i--)
      b = (b<<WORD) | (a->dp[i]);
    
  return b;
}

void Polynomial::crt(){
  callCRT(this->bn_coefs[0],
          this->bn_coefs.size(),
          this->get_device_crt_residues(),
          this->get_crt_spacing(),
          Polynomial::CRTPrimes.size(),
          this->get_stream()
    );
  this->set_crt_computed(true);
}

void Polynomial::icrt(){  
  cudaError_t result;

  // Adjusts bn_coefs size
  int diff = (int)bn_coefs.size() - this->get_crt_spacing();
  if(diff < 0){
    diff = -diff;

    // Adds more coefficients
    bn_t *coefs;
    
    // Doing this we need only one call to cudaMallocManaged
    result = cudaMallocManaged(&coefs, diff*sizeof(bn_t));
    assert(result == cudaSuccess);

    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
    for(int i =0; i < diff; i++){
      bn_new(&coefs[i]);
      bn_coefs.push_back(&coefs[i]);
    }
  }else if(diff > 0){

      // Releases unused coefficients
      for(int i = 0; i < diff;i++){
        bn_t *coef = bn_coefs.back();
        bn_coefs.pop_back();

        bn_free(coef);
        result = cudaFree(coef);
        assert(result == cudaSuccess);
      }
  }

  //////////////////////////////////////////////////
  // At this point, bn_coefs.size() == CRTSPACING //
  //////////////////////////////////////////////////
  callICRT( this->bn_coefs[0],
        this->get_device_crt_residues(),
        this->get_crt_spacing(),
        Polynomial::CRTPrimes.size(),
        this->get_stream()
      );

  this->set_icrt_computed(true);
}

void Polynomial::update_device_data(){
  /**
   * This function copy the polynomial to GPU's memory in bn_t format
   */

  if(this->get_device_updated())
    return;

  cudaError_t result;
  #if defined(VERBOSE) || defined(VERBOSEMEMORYCOPY) 
  std::cout << "Copying data to GPU." << std::endl;
  #endif


  this->ON_COPY = true
;  // Updated CRTSPACING    
  // Verifica se o espaçamento é válido. Se não for, ajusta.
  if(this->get_crt_spacing() < this->deg()+1){
    const int new_spacing = this->deg()+1;

    // Data on device isn't updated (we check it on begginning)
    // So, update_crt_spacing(int) will only update CRTSpacing and alloc memory
    this->update_crt_spacing(new_spacing);
  }

  result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);

  // Adjusts bn_coefs size
  int diff = (int)bn_coefs.size() - this->get_crt_spacing();
  if(diff < 0){
    diff = -diff;
    // std::cout << "Smaller" << std::endl;
    // Adds more coefficients
    bn_t *coefs;
    
    // Doing this we need only one call to cudaMallocManaged
    result = cudaMallocManaged(&coefs, diff*sizeof(bn_t));
    assert(result == cudaSuccess);

    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
    for(int i =0; i < diff; i++){
      // bn_new(&coefs[i]);
      bn_coefs.push_back(&coefs[i]);
    }
  }else if(diff > 0){
      // std::cout << "Bigger" << std::endl;

      // Releases unused coefficients
      for(int i = 0; i < diff;i++){
        bn_t *coef = bn_coefs.back();
        bn_coefs.pop_back();

        bn_free(coef);
        result = cudaFree(coef);
        assert(result == cudaSuccess);
      }
  }
  
  /////////////////////////////////////////////////
  // At this point bn_coefs.size() == CRTSPACING //
  /////////////////////////////////////////////////

  /**
   * Converts ZZs to bn_t
   */

  for(int i = 0; i < bn_coefs.size(); i++){
    if(bn_coefs[i] == NULL)
      std::cout << "Achei!" << std::endl;
    get_words(bn_coefs[i],get_coeff(i));
  }

  result = cudaMemsetAsync(this->get_device_crt_residues(),
                  0,
                  this->get_crt_spacing()*Polynomial::CRTPrimes.size()*sizeof(cuyasheint_t),
                  this->get_stream());
  assert(result == cudaSuccess);

  /**
  *  CRT
  */
  this->crt();

  /**
   * Releases bn_t
   */
  
  for(int i = 0; i < bn_coefs.size(); i++){
    bn_free(bn_coefs[i]);
  }
  cudaFree(bn_coefs[0]);
  bn_coefs.clear();

  result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);

  this->ON_COPY = false;
  this->set_device_updated(true);

}

void Polynomial::update_host_data(){
  
    if(get_host_updated())
      return;
    if(!get_icrt_computed())
      this->icrt();

    #ifdef VERBOSE
    std::cout << "Copying data to CPU." << std::endl;
    #endif

    assert(Polynomial::CRTPrimes.size() > 0);

    // Prepare this polynomial to receive this->deg()+1 coefficients
    if((int)coefs.size() <= this->deg())
      set_coeffs(this->deg()+1);

    cudaError_t result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    /**
     * Convert bn_t to ZZ
     */
    this->set_coeffs(this->deg()+1);
    for(int i = 0; i <= this->deg(); i++){
      // ZZ coef = get_ZZ(bn_coefs[i]);
      ZZ coef = get_ZZ(bn_coefs[i])% Polynomial::CRTProduct;
      this->set_coeff(i,coef);
    }

    /**
    * Releases bn_t
    */

    for(int i = 0; i < bn_coefs.size(); i++){
      bn_free(bn_coefs[i]);
    }
    cudaFree(bn_coefs[0]);
    bn_coefs.clear();
    
    this->set_host_updated(true);

}

void Polynomial::DivRem(Polynomial a,Polynomial b,Polynomial &quot,Polynomial &rem){
  // Returns x = a % b

  if(a.get_host_updated() == false && b.get_host_updated() == false){
    // Operates on GPU
    throw "DivRem for GPU not implemented yet";
  }else{
    if(!a.get_host_updated())
        a.update_host_data();

    if(!b.get_host_updated())
        b.update_host_data();
    
    // No need to reduce
    if(a.deg() <= 0)
      return;
    if(check_special_rem_format(b)){
      #ifdef VERBOSE
      std::cout << "Rem in special mode."<<std::endl;
      #endif

      const unsigned int half = b.deg()-1;     

      rem.set_coeffs(half+1);
      #pragma omp parallel for
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
