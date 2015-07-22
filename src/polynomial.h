#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H
#include <stdio.h>
#include <vector>
#include <NTL/ZZ.h>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include "common.h"

NTL_CLIENT

class Polynomial{
  public:
    // Attributes
    int CRTSPACING;// Stores the distance between the zero-coeff of two consecutive residues in d_polyCRT
    static ZZ CRTProduct;
    static std::vector<long> CRTPrimes;
    static Polynomial *global_phi;
    static ZZ global_mod;

    // Constructors
    Polynomial(){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      if(this->global_mod > 0){
        // If a global mod is defined, use it
        this->mod = this->global_mod; // Doesn't copy. Uses the reference.
      }
      if(this->global_phi){
        // If a global phi is defined, use it
        this->phi = this->global_phi; // Doesn't copy. Uses the reference.
      }
      // CRT Spacing not set
      this->CRTSPACING = -1;
      coefs.push_back(ZZ(0));

      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Polynomial(ZZ p){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->mod = ZZ(p);// Copy

      if(this->global_phi){
      //   // If a global phi is defined, use it
        this->phi = this->global_phi; // Doesn't copy. Uses the reference.
      }
      // CRT Spacing not set
      this->CRTSPACING = -1;

      coefs.push_back(ZZ(0));
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod << " but no phi."<< std::endl;
      #endif
    }
    Polynomial(ZZ p,Polynomial P){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->mod = ZZ(p);// Copy
      *(this->phi) = Polynomial(P);// Copy

      // CRT Spacing set to two times phi degree+1
      this->CRTSPACING = 2*(this->phi->deg())+1;

      coefs.push_back(ZZ(0));
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this-> phi << std::endl;
      #endif
    }
    Polynomial(int spacing){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      if(this->global_mod > 0){
        // If a global mod is defined, use it
        this->mod = this->global_mod; // Doesn't copy. Uses the reference.
      }
      if(this->global_phi){
      //   // If a global phi is defined, use it
        this->phi = this->global_phi; // Doesn't copy. Uses the reference.
      }
      // CRT Spacing set to spacing
      this->CRTSPACING = spacing;
      coefs.push_back(ZZ(0));

      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Polynomial(ZZ p,Polynomial P,int spacing){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->mod = ZZ(p);// Copy
      *(this->phi) = Polynomial(P);// Copy

      // CRT Spacing set to spacing
      this->CRTSPACING = spacing;

      coefs.push_back(ZZ(0));
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this-> phi << std::endl;
      #endif
    }
    Polynomial(ZZ p,int spacing){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->mod = ZZ(p);// Copy

      // CRT Spacing set to spacing
      this->CRTSPACING = spacing;

      coefs.push_back(ZZ(0));
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod << std::endl;
      #endif
    }
    // Functions and methods
    // Operators
    Polynomial operator=(Polynomial b){//Copy
        this->CRTSPACING = b.CRTSPACING;
        this->set_host_updated(b.get_host_updated());
        this->set_device_updated(b.get_device_updated());
        this->set_coeffs(b.get_coeffs());
        this->polyCRT = b.get_crt_residues();
        this->d_polyCRT = b.get_device_crt_residues();
        this->set_mod(b.get_mod());
        this->expected_degree = b.deg();
        // this->set_phi(b.get_phi());

        #ifdef VERBOSE
          std::cout << "Polynomial copied. " << std::endl;
        #endif
        return *this;
    }
    Polynomial operator+(Polynomial b){
      // Check align
      if(this->CRTSPACING != b.CRTSPACING){
        int new_spacing = max(this->CRTSPACING,b.CRTSPACING);
        this->update_crt_spacing(new_spacing);
        b.update_crt_spacing(new_spacing);
      }

      #ifdef VERBOSE
      std::cout << "Adding:" << std::endl;
      std::cout << "this: " << *this << std::endl;
      std::cout << "other " << other << std::endl;
      #endif

      // Apply CRT and copy data to global memory, if needed
      #pragma omp parallel sections
      {
          #pragma omp section
          {

              if(!this->get_device_updated()){
                this->crt();
                this->update_device_data();
              }

          }
          #pragma omp section
          {
              if(!b.get_device_updated()){
                  b.crt();
                  b.update_device_data();
              }
          }
      }


    long *d_result = CUDAFunctions::callPolynomialAdd(this->stream,this->get_device_crt_residues(),b.get_device_crt_residues(),(int)(this->CRTSPACING*this->polyCRT.size()));

    Polynomial c(this->get_mod(),this->get_phi(),this->CRTSPACING);
    c.set_device_crt_residues(d_result);

    cudaDeviceSynchronize();
    return c;
    }
    Polynomial operator+=(Polynomial b){
      return (*this)+b;
    }
    Polynomial operator-(Polynomial b){
      // To-do
      throw "Not implemented!";
    }
    Polynomial operator-=(Polynomial b){
      return (*this)-b;
    }
    Polynomial operator*(Polynomial b){
      // To-do
      throw "Not implemented!";
    }
    Polynomial operator*=(Polynomial b){
      return (*this)*b;
    }
    Polynomial operator/(Polynomial b){
      Polynomial quot;
      Polynomial rem;

      #pragma omp sections
      {
          #pragma omp section
          {
            if(!this->get_host_updated())
              this->icrt();
          }
          #pragma omp section
          {
            if(!b.get_host_updated())
              b.icrt();
          }
      }

      Polynomial::DivRem((*this),b,(&quot),(&rem));
      return quot;
    }
    Polynomial operator/=(Polynomial b){
      return (*this)/b;
    }

    Polynomial operator+(ZZ b){
      Polynomial p(*this);
      if(!p.get_host_updated())
        p.icrt();
      p.set_coeff(0,p.get_coeff(0)+b);
      return p;
    }
    Polynomial operator+=(ZZ b){
      return (*this)+b;
    }
    Polynomial operator-(ZZ b){
      Polynomial p(*this);
      if(!p.get_host_updated())
        p.icrt();
      p.set_coeff(0,p.get_coeff(0)-b);
      return p;
    }
    Polynomial operator-=(ZZ b){
      return (*this)-b;
    }
    Polynomial operator*(ZZ b){
      Polynomial p(*this);
      if(!p.get_host_updated())
        p.icrt();

      #pragma omp for
      for(int i = 0; i <= p.deg();i++)
        p.set_coeff(i,conv<ZZ>(p.get_coeff(i)*b));
      return p;
    }
    Polynomial operator*=(ZZ b){
      return (*this)*b;
    }
    Polynomial operator/(ZZ b){
      #ifdef VERBOSE
        std::cout << "This operations is slower than a division by an integer." << std::endl;
      #endif
      Polynomial p(*this);
      if(!p.get_host_updated())
        p.icrt();

      #pragma omp for
      for(int i = 0; i <= p.deg();i++)
        p.set_coeff(i,p.get_coeff(i)/b);

      return p;
    }
    Polynomial operator/=(ZZ b){
      return (*this)/b;
    }
    Polynomial operator+(int b){
      if(!this->get_host_updated()){
        // If host data isn't updated, it is faster to convert b to Polynomial and operates on GPU
        // Polynomial other(this->get_mod(),this->get_phi(),this->get_crt_spacing());
        Polynomial other(this->get_mod(),this->get_crt_spacing());
        return (*this) + b;
      }else{
        Polynomial p(*this);

        p.set_coeff(0,conv<ZZ>(p.get_coeff(0)+b));
        return p;
      }

    }
    Polynomial operator+=(int b){
      return (*this)+b;
    }
    Polynomial operator-(int b){
      if(!this->get_host_updated()){
        // If host data isn't updated, it is faster to convert b to Polynomial and operates on GPU
        // Polynomial other(this->get_mod(),this->get_phi(),this->get_crt_spacing());
        Polynomial other(this->get_mod(),this->get_crt_spacing());
        return (*this) - b;
      }else{
        Polynomial p(*this);

        p.set_coeff(0,conv<ZZ>(p.get_coeff(0)-b));
        return p;
      }

    }
    Polynomial operator-=(int b){
      return (*this)-b;
    }
    Polynomial operator*(int b){
      if(!this->get_host_updated()){
        // If host data isn't updated, it is faster to operates on GPU
        // To-do
        throw "Not implemented!";
      }else{
        Polynomial p(*this);

        #pragma omp for
        for(int i = 0; i <= p.deg();i++)
          p.set_coeff(i,conv<ZZ>(p.get_coeff(i)*b));
        return p;
      }
    }
    Polynomial operator*=(int b){
      return (*this)+b;
    }
    Polynomial operator/(int b){
      if(!this->get_host_updated()){
        // If host data isn't updated, it is faster to operates on GPU
        // To-do
        throw "Not implemented!";
      }else{
        Polynomial p(*this);

        #pragma omp for
        for(int i = 0; i <= p.deg();i++)
          p.set_coeff(i,conv<ZZ>(p.get_coeff(i)/b));
        return p;
      }
    }
    Polynomial operator/=(int b){
      return (*this)+b;
    }

    Polynomial operator%(Polynomial b){
      Polynomial quot;
      Polynomial rem;

      #pragma omp sections
      {
          #pragma omp section
          {
            if(!this->get_host_updated())
              this->icrt();
          }
          #pragma omp section
          {
            if(!b.get_host_updated())
              b.icrt();
          }
      }

      Polynomial::DivRem((*this),b,(&quot),(&rem));
      return rem;
    }
    Polynomial operator%=(Polynomial b){
      return *this%b;
    }
    bool operator==(Polynomial b){
      // To-do
      throw "Not implemented!";
    }
    bool operator!=(Polynomial b){
      return !((*this) == b);
    }


    // gets and sets
    ZZ get_mod(){
      // Returns a copy of mod
      return ZZ(this->mod);
    }
    void set_mod(ZZ value){
      this->mod = ZZ(value);
      #ifdef VERBOSE
        std::cout << "Polynomial mod set to " << this->mod << std::endl;
      #endif
    }
    Polynomial get_phi(){
      // Returns a copy of phi
      return *(this->phi);
    }
    void set_phi(Polynomial p){
      *(this->phi) = Polynomial(p);
      #ifdef VERBOSE
        std::cout << "Polynomial phi set to " << this->phi << std::endl;
      #endif
    }


    ZZ get_coeff(int index){
      // Returns a copy of coefficient at this index
      if(index > this->deg())
        return conv<ZZ>(0);
      else
        return this->coefs.at(index);
    }
    void set_coeff(int index,ZZ value){
      if((unsigned int)(index) >= this->coefs.size()){
        #if VERBOSE
          std::cout << "Resizing from "<< this->coefs.size() << " to " << index+1 << std::endl;
        #endif

        this->coefs.resize(index+1);
      }
      std::vector<ZZ>::iterator it = this->coefs.begin();
      this->coefs.insert(it+index,value);

      this->expected_degree = this->coefs.size()-1;
      #ifdef VERBOSE
        std::cout << "Polynomial coeff " << index << " set to " << this->coefs[index] << std::endl;
      #endif
    }
    void set_coeff(int index,int value){
      this->set_coeff(index,ZZ(value));
    }
    std::vector<ZZ> get_coeffs(){
      // Returns a copy of all coefficients
      std::vector<ZZ> coefs_copy(this->coefs);
      return coefs_copy;
    }
    void set_coeffs(std::vector<ZZ> values){
      // Replaces all coefficients
      this->coefs = values;
      this->expected_degree = this->coefs.size()-1;

    }
    std::vector<std::vector<long> > get_crt_residues(){
      std::vector<std::vector<long> > crt_residues_copy(this->polyCRT);
      return crt_residues_copy;
    }
    long* get_device_crt_residues(){
      // Returns the address of crt residues at device memory
      return this->d_polyCRT;
    }
    void set_device_crt_residues(long *residues){
      this->d_polyCRT = residues;
    }

    void crt();
    void icrt();
    int get_crt_spacing(){
      return this->CRTSPACING;
    }
    static void gen_crt_primes(ZZ q,long degree){
        // We will use 63bit primes to fit long data type (64 bits raises "GenPrime: length too large")
        ZZ M = ZZ(1);
        std::vector<long> P;

        // long q_size = conv<long>(NTL::NumBytes(q))*8;
        int primes_size = 63;
        // long nprimes = (degree*2*q_size)/primes_size+1;
        long n;

        while( (M < degree*q*q) ){
            n = NTL::GenPrime_long(primes_size);
            P.push_back(n);
            M *=(n);
        }

        Polynomial::CRTProduct = M;
        Polynomial::CRTPrimes = P;

        std::cout << P.size() << " primes generated." << std::endl;
        #ifdef DEBUG
        std::cout << "Primes set - M:" << Polynomial::CRTProduct << std::endl;
        // std::cout << "Primes: "<< Polynomial::CRTPrimes << std::endl;
        #endif
    }


    void update_device_data();
    void set_device_updated(bool b){
      this->DEVICE_IS_UPDATE = b;
    }
    bool get_device_updated(){
      return this->DEVICE_IS_UPDATE;
    }
    void update_host_data();
    void set_host_updated(bool b){
      this->HOST_IS_UPDATED = b;
    }
    bool get_host_updated(){
      return this->HOST_IS_UPDATED;
    }


    int deg(){
      return this->coefs.size()-1;
    }
    ZZ lead_coeff(){
      if(this->deg() >= 0){
        return this->get_coeff(this->deg());
      }else{
        return ZZ(0);
      }
    }
    static bool check_special_rem_format(Polynomial p){
      if(p.get_coeff(0) != 1){
        return false;
      }
      if(p.lead_coeff() != ZZ(1)){
        return false;
      }
      for(int i =1;i < p.deg();i++){
        if(p.get_coeff(i) != 0)
          return false;
      }
      return true;
    }
    void update_crt_spacing(int new_spacing){
      if(this->CRTSPACING == new_spacing)
        return;

      // If updated data lies in gpu's global memory, realign it
      if(this->get_device_updated()){
        long * d_pointer = CUDAFunctions::callRealignCRTResidues(this->stream, this->CRTSPACING,new_spacing,this->get_device_crt_residues(),this->deg()+1,Polynomial::CRTPrimes.size());
        if(d_pointer != NULL){
          this->set_device_crt_residues(d_pointer);
        }else{
          #ifdef VERBOSE
          std::cout << "Old spacing is equal new spacing." << std::endl;
          #endif
        }
      }
      this->CRTSPACING = new_spacing;

    }
    int get_expected_degre(){
      return expected_degree;
    }
    static void DivRem(Polynomial a,Polynomial b,Polynomial *quot,Polynomial *rem);
    static Polynomial InvMod(Polynomial a,Polynomial b){
      // To-do
      throw "Not implemented!";
    }
    static void BuildNthCyclotomic(Polynomial *phi, int n){
      std::vector<Polynomial> aux_phi( n+1);

      for (long i = 1; i <= n; i++) {
         Polynomial t;
         t.set_coeff(0,ZZ(1));

         for (long j = 1; j <= i-1; j++)
            if (i % j == 0)
               t *= aux_phi[j];

         Polynomial mono;
         mono.set_coeff(i,ZZ(1));
         aux_phi[i] = (mono - 1)/t;

        //  cout << aux_phi[i] << "\n";
      }
       *phi = aux_phi[n];
    }
    static void random(Polynomial *a,int degree){
      if(a->global_mod > 0)
        for(int i = 0;i <= degree;i++)
          a->set_coeff(i,ZZ(rand()));
      else
        for(int i = 0;i <= degree;i++)
          a->set_coeff(i,ZZ(rand()) % a->global_mod);
    }
  private:
    // Attributes
    cudaStream_t stream;
    int expected_degree; // This variable stores the expected degree for this polinomial
    std::vector<ZZ> coefs;
    std::vector<std::vector<long> > polyCRT; // Must be initialized by crt()
    long *d_polyCRT; // Must be initialized on CRTSPACING definition and updated by crt(), if needed
    ZZ mod;
    Polynomial *phi;

    bool ON_COPY;
    bool HOST_IS_UPDATED;
    bool DEVICE_IS_UPDATE;
    //Functions and methods

};
#endif
