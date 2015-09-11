#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H
#include <stdio.h>
#include <vector>
#include <sstream>
#include <NTL/ZZ.h>
#include <NTL/ZZ_pEX.h>
#include <NTL/ZZ_pXFactoring.h>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include "common.h"
#include <algorithm>

NTL_CLIENT

uint64_t polynomial_get_cycles();

class Polynomial{
  public:
    // Attributes
    int CRTSPACING =-1;// Stores the distance between the zero-coeff of two consecutive residues in d_polyCRT
    static ZZ CRTProduct;
    static std::vector<cuyasheint_t> CRTPrimes;
    static Polynomial *global_phi;
    static ZZ global_mod;
    static bool phi_set;

    // Constructors
    Polynomial(){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      if(Polynomial::global_mod > 0)
        // If a global mod is defined, use it
        this->mod = Polynomial::global_mod; // Doesn't copy. Uses the reference.
      
      if(Polynomial::global_phi){
        // If a global phi is defined, use it
        this->phi = Polynomial::global_phi; // Doesn't copy. Uses the reference.

        // If the irreductible polynomial have degree N, this polynomial's degree will be limited to N-1
        this->CRTSPACING = Polynomial::global_phi->deg();
      }else
        // CRT Spacing not set
        this->CRTSPACING = -1;
      

      if(Polynomial::phi_set)
        this->coefs.resize(this->get_phi().deg()+1);
      

      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Polynomial(ZZ p){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      this->mod = ZZ(p);// Copy

      if(Polynomial::global_phi){
      //   // If a global phi is defined, use it
        this->phi = Polynomial::global_phi; // Doesn't copy. Uses the reference.
      }
      // CRT Spacing not set
      this->CRTSPACING = -1;

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod << " but no phi."<< std::endl;
      #endif
    }
    Polynomial(ZZ p,Polynomial P){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      this->mod = ZZ(p);// Copy
      *(this->phi) = Polynomial(P);// Copy

      // CRT Spacing should store the expected number of coefficients
      // If the irreductible polynomial have degree N, this polynomial's degree will be limited to N-1
      this->CRTSPACING = this->phi->deg();

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this-> phi << std::endl;
      #endif
    }
    Polynomial(int spacing){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      if(Polynomial::global_mod > 0){
        // If a global mod is defined, use it
        this->mod = Polynomial::global_mod; // Doesn't copy. Uses the reference.
      }
      if(Polynomial::global_phi){
      //   // If a global phi is defined, use it
        this->phi = Polynomial::global_phi; // Doesn't copy. Uses the reference.
      }
      // CRT Spacing set to spacing
      this->CRTSPACING = spacing;
      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }

      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Polynomial(ZZ p,Polynomial P,int spacing){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      this->mod = ZZ(p);// Copy
      this->phi = &P;// Copy

      // std::cout << this->get_phi().to_string() << std::endl;

      // CRT Spacing set to spacing
      this->CRTSPACING = spacing;

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this->phi << std::endl;
      #endif
    }
    Polynomial(ZZ p,int spacing){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      this->mod = ZZ(p);// Copy

      // CRT Spacing set to spacing
      this->CRTSPACING = spacing;

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod << std::endl;
      #endif
    }
    Polynomial(Polynomial *b){
      // Copy
      this->copy(*b);
    }

    void copy(Polynomial b){
      #ifdef VERBOSE
      uint64_t start,stop;
      start = polynomial_get_cycles();
      #endif


      this->CRTSPACING = b.CRTSPACING;
      this->set_host_updated(b.get_host_updated());
      this->set_device_updated(b.get_device_updated());

      this->polyCRT = b.get_crt_residues();

      if(b.get_host_updated())
        this->set_coeffs(b.get_coeffs());

      this->d_polyCRT = b.get_device_crt_residues();

      // The line below takes 66k cycles to return. We will comment it for now.
      // this->set_mod(b.get_mod());
      
      // if(this != Polynomial::global_phi){
        // The line below takes 1M cycles to return. We will comment it for now.
        // this->set_phi(b.get_phi());
      // }

      this->expected_degree = b.get_expected_degre();
      this->set_crt_computed(b.get_crt_computed());
      this->set_icrt_computed(b.get_icrt_computed());

      #ifdef VERBOSE
      stop = polynomial_get_cycles();
      std::cout << (stop-start) << " cycles to copy" << std::endl;
      #endif
    }
    // Functions and methods
    // Operators
    // std::ostream &operator<<(std::ostream &os, Polynomial &m) {
    //   for(int i = 0; i <=  m.deg();i++)
    //     os << m.get_coeff(i);
    //   return os;
    // }

    std::string to_string(){
      if(!this->get_icrt_computed()){
        this->icrt();
      }

      stringstream ss;
      for(int i = 0; i <=  this->deg();i++)
        ss << this->get_coeff(i) << ", ";
      return ss.str();;
    }
    Polynomial operator=(Polynomial b){//Copy
        this->copy(&b);

        #ifdef VERBOSE
          std::cout << "Polynomial copied. " << std::endl;
        #endif
        return *this;
    }
    Polynomial operator+(Polynomial b){
      #ifdef ADDONCPUINPOSSIBLE
      if(!this->get_device_updated() && !b.get_device_updated()){
        // CPU add
        #ifdef VERBOSE
        std::cout << "Operator+ on CPU" << std::endl;
        #endif
        Polynomial c(*this);
        c.CPUAddition(&b);
        return c;
      }else{
      #endif
        #ifdef VERBOSE
        std::cout << "Operator+ on GPU" << std::endl;
        #endif
        // Check align
        if(this->CRTSPACING != b.CRTSPACING){
          int new_spacing = std::max(this->CRTSPACING,b.CRTSPACING);
          this->update_crt_spacing(new_spacing);
          b.update_crt_spacing(new_spacing);
        }

        #ifdef VERBOSE
        std::cout << "Adding:" << std::endl;
        // std::cout << "this: " << this->to_string() << std::endl;
        // std::cout << "other " << b.to_string() << std::endl;
        #endif

        // Apply CRT and copy data to global memory, if needed
        #pragma omp parallel sections num_threads(2)
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


        cuyasheint_t *d_result = CUDAFunctions::callPolynomialAddSub(this->stream,this->get_device_crt_residues(),b.get_device_crt_residues(),(int)(this->CRTSPACING*Polynomial::CRTPrimes.size()),ADD);

        Polynomial c(this->get_mod(),this->get_phi(),this->CRTSPACING);
        c.set_device_crt_residues(d_result);
        c.set_host_updated(false);
        c.set_device_updated(true);
        cudaDeviceSynchronize();
        return c;
      #ifdef ADDONCPUINPOSSIBLE
      }
      #endif
    }
    Polynomial operator+=(Polynomial b){
      #ifdef ADDONCPUINPOSSIBLE
      if(!this->get_device_updated() && !b.get_device_updated()){
        // CPU add
        #ifdef VERBOSE
        std::cout << "Operator+ on CPU" << std::endl;
        #endif
        this->CPUAddition(&b);
        return *this;
      }else{
      #endif
        #ifdef VERBOSE
        std::cout << "Operator+ on GPU" << std::endl;
        #endif
        // Check align
        if(this->CRTSPACING != b.CRTSPACING){
          int new_spacing = std::max(this->CRTSPACING,b.CRTSPACING);
          this->update_crt_spacing(new_spacing);
          b.update_crt_spacing(new_spacing);
        }

        #ifdef VERBOSE
        std::cout << "Adding:" << std::endl;
        // std::cout << "this: " << this->to_string() << std::endl;
        // std::cout << "other " << b.to_string() << std::endl;
        #endif

        // Apply CRT and copy data to global memory, if needed
        #pragma omp parallel sections num_threads(2)
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


        CUDAFunctions::callPolynomialAddSubInPlace(this->stream,this->get_device_crt_residues(),b.get_device_crt_residues(),(int)(this->CRTSPACING*Polynomial::CRTPrimes.size()),ADD);

        this->set_host_updated(false);
        this->set_device_updated(true);
        cudaDeviceSynchronize();
      #ifdef ADDONCPUINPOSSIBLE
      }
      #endif
      return *this;
    }
    Polynomial operator-(Polynomial b){
      // Check align
      if(this->CRTSPACING != b.CRTSPACING){
        int new_spacing = std::max(this->CRTSPACING,b.CRTSPACING);
        this->update_crt_spacing(new_spacing);
        b.update_crt_spacing(new_spacing);
      }

      #ifdef VERBOSE
      std::cout << "Sub:" << std::endl;
      // std::cout << "this: " << this->to_string() << std::endl;
      // std::cout << "other " << b.to_string() << std::endl;
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

      cuyasheint_t *d_result = CUDAFunctions::callPolynomialAddSub(this->stream,this->get_device_crt_residues(),b.get_device_crt_residues(),(int)(this->CRTSPACING*this->polyCRT.size()),SUB);

      Polynomial c(this->get_mod(),this->get_phi(),this->CRTSPACING);
      c.set_device_crt_residues(d_result);
      c.set_host_updated(false);
      c.set_device_updated(true);
      cudaDeviceSynchronize();
      return c;
    }
    Polynomial operator-=(Polynomial b){
      this->set_device_crt_residues( ((*this)-b).get_device_crt_residues());
      return *this;
    }
    Polynomial operator*(Polynomial b){

      // Check align
      int new_spacing = pow(2,ceil(log2(this->deg()+b.deg())));
        if(new_spacing < CUDAFunctions::N)
          new_spacing = CUDAFunctions::N;
        else if(new_spacing != CUDAFunctions::N){
          // Re-compute W matrix
          CUDAFunctions::init(new_spacing);
      }
      this->update_crt_spacing(new_spacing);
      b.update_crt_spacing(new_spacing);

      #ifdef VERBOSE
      std::cout << "Mul:" << std::endl;
      // std::cout << "this: " << this->to_string() << std::endl;
      // std::cout << "other " << b.to_string() << std::endl;
      #endif

      // Apply CRT and copy data to global memory, if needed
      #pragma omp sections
      {
          #pragma omp section
          {

              if(!this->get_device_updated()){
                this->crt();
                this->update_device_data(2);
              }

          }
          #pragma omp section
          {
              if(!b.get_device_updated()){
                  b.crt();
                  b.update_device_data(2);
              }
          }
      }

      cuyasheint_t *d_result = CUDAFunctions::callPolynomialMul(this->stream,this->get_device_crt_residues(),b.get_device_crt_residues(),this->CRTSPACING,this->CRTPrimes.size());

      Polynomial c(this->get_mod(),this->get_phi(),this->CRTSPACING);
      c.set_device_crt_residues(d_result);
      c.set_host_updated(false);
      c.set_device_updated(true);
      // cudaDeviceSynchronize();
      return c;
    }
    Polynomial operator*=(Polynomial b){
      this->set_device_crt_residues( ((*this)*b).get_device_crt_residues());
      return *this;
    }
    Polynomial operator/(Polynomial b){
      Polynomial quot;
      Polynomial rem;

      // #pragma omp parallel sections
      // {
      //     #pragma omp section
      //     {
      //       if(!this->get_host_updated())
      //         this->icrt();
      //     }
      //     #pragma omp section
      //     {
      //       if(!b.get_host_updated())
      //         b.icrt();
      //     }
      // }

      Polynomial::DivRem((*this),b,quot, rem);
      return quot;
    }
    Polynomial operator/=(Polynomial b){
      this->set_device_crt_residues( ((*this)/b).get_device_crt_residues());
      return *this;
    }
    Polynomial operator%(Polynomial b){
      Polynomial quot;
      Polynomial rem;

      // #pragma omp parallel sections
      // {
      //     #pragma omp section
      //     {
      //       if(!this->get_host_updated())
      //         this->icrt();
      //     }
      //     #pragma omp section
      //     {
      //       if(!b.get_host_updated())
      //         b.icrt();
      //     }
      // }

      Polynomial::DivRem((*this),b,quot, rem);
      // rem.icrt();
      // std::cout << rem.to_string() << std::endl;
      return rem;
    }
    Polynomial operator%=(Polynomial b){
      this->set_device_crt_residues( ((*this)%b).get_device_crt_residues());
      return *this;
    }

    Polynomial operator+(ZZ b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        // Convert to polynomial and send to device
        Polynomial B(this->get_mod(),this->get_phi(),this->get_crt_spacing());
        B.set_coeff(0,b);
        return p+B;
      }

      p.set_coeff(0,p.get_coeff(0)+b);
      p.set_device_updated(false);
      return p;
    }
    Polynomial operator+=(ZZ b){
      this->copy(((*this)+b));
      return *this;
    }
    Polynomial operator-(ZZ b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        // Convert to polynomial and send to device
        Polynomial B(this->get_mod(),this->get_phi(),this->get_crt_spacing());
        B.set_coeff(0,b);
        return p-B;
      }
      p.set_coeff(0,p.get_coeff(0)-b);
      p.set_device_updated(false);
      return p;
    }
    Polynomial operator-=(ZZ b){
      this->copy(((*this)-b));
      return *this;
    }
    Polynomial operator*(ZZ b){
      Polynomial p(*this);
      if(!p.get_host_updated())
        // We cannot store ZZ integers in device's memory
        p.icrt();


      //#pragma omp parallel for
      for(int i = 0; i <= p.deg();i++)
        p.set_coeff(i,p.get_coeff(i)*b);
      p.set_device_updated(false);
      return p;
    }
    Polynomial operator*=(ZZ b){
      this->copy(((*this)*b));
      return *this;
    }
    Polynomial operator%(ZZ b){
      Polynomial p(*this);

      if(!p.get_host_updated()){
        //
        p.icrt();
      }

      // #pragma omp parallel for
      for(int i = 0; i <= p.deg();i++){
        // ZZ value = p.get_coeff(i)%b;
        // std::cout << "value: " << value << std::endl << "b: " << b<< std::endl;
        // p.set_coeff(i,value);
        p.set_coeff(i,p.get_coeff(i)%b);
      }
      p.set_device_updated(false);

      return p;
    }
    Polynomial operator%=(ZZ b){
      this->copy(((*this)%b));
      return *this;
    }
    Polynomial operator/(ZZ b){
      Polynomial p(*this);
      if(!p.get_host_updated()){
        #warning "Polynomial division on device not implemented!";
        p.icrt();
      }

      // #pragma omp parallel for
      for(int i = 0; i <= p.deg();i++)
        p.set_coeff(i,p.get_coeff(i)/b);
      p.set_device_updated(false);

      return p;
    }
    Polynomial operator/=(ZZ b){
      this->copy(((*this)/b));
      return *this;
    }
    Polynomial operator+(uint64_t b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        CUDAFunctions::callPolynomialOPInteger(ADD,this->stream,p.get_device_crt_residues(),b,p.CRTSPACING,Polynomial::CRTPrimes.size());
        p.set_device_updated(true);
        p.set_host_updated(false);
        return p;
      }else{
        return (*this)+ZZ(b);
      }
    }
    Polynomial operator+=(uint64_t b){
      this->set_device_crt_residues( ((*this)+b).get_device_crt_residues());
      return *this;
    }
    Polynomial operator-(uint64_t b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        CUDAFunctions::callPolynomialOPInteger(SUB,this->stream,p.get_device_crt_residues(),b,p.CRTSPACING,Polynomial::CRTPrimes.size());
        p.set_device_updated(true);
        p.set_host_updated(false);
        return p;
      }else{
        return (*this)-ZZ(b);
      }
    }
    Polynomial operator-=(uint64_t b){
      this->set_device_crt_residues( ((*this)-b).get_device_crt_residues());
      return *this;
    }
    Polynomial operator*(uint64_t b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        CUDAFunctions::callPolynomialOPInteger(MUL,this->stream,p.get_device_crt_residues(),b,p.CRTSPACING,Polynomial::CRTPrimes.size());
        p.set_device_updated(true);
        p.set_host_updated(false);
        return p;
      }else{
        return (*this)*ZZ(b);
      }
    }
    Polynomial operator*=(uint64_t b){
      this->set_device_crt_residues( ((*this)*conv<ZZ>(b)).get_device_crt_residues());
      return *this;
    }
    Polynomial operator/(uint64_t b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        CUDAFunctions::callPolynomialOPInteger(DIV,this->stream,p.get_device_crt_residues(),b,p.CRTSPACING,Polynomial::CRTPrimes.size());
        p.set_device_updated(true);
        p.set_host_updated(false);
        return p;
      }else{
        return (*this)/ZZ(b);
      }
    }
    Polynomial operator%(uint64_t b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        CUDAFunctions::callPolynomialOPInteger(MOD,this->stream,p.get_device_crt_residues(),b,p.CRTSPACING,Polynomial::CRTPrimes.size());
        p.set_device_updated(true);
        p.set_host_updated(false);
        return p;
      }else{
        return (*this)%ZZ(b);
      }
    }
    Polynomial operator/=(uint64_t b){
      this->set_device_crt_residues( ((*this)/conv<ZZ>(b)).get_device_crt_residues());
      return *this;
    }

  bool operator==(Polynomial b){
      if(!this->get_host_updated()){
          this->icrt();
      }
      if(!b.get_host_updated()){
        b.icrt();
      }
      this->normalize();
      b.normalize();

      if(this->deg() != b.deg())
        return false;

      for( int i = 0; i <= this->deg();i++){
        if(this->get_coeff(i) != b.get_coeff(i))
          return false;
      }
      return true;
    }
    bool operator!=(Polynomial b){
      return !((*this) == b);
    }

    void MulMod(ZZ b,ZZ mod){
      if(!this->get_host_updated())
        // We cannot store ZZ integers in device's memory
        this->icrt();

      // #pragma omp parallel for
      for(int i = 0; i <= this->deg();i++)
        this->set_coeff(i,NTL::MulMod(this->get_coeff(i),b,mod));
      this->set_device_updated(false);
    }
    void CPUAddition(Polynomial *b){
      // Forces the addition to be executed by CPU
      // This method supposes that there is no need to apply CRT/ICRT on operands
      for( unsigned int i = 0; i <= std::max(this->deg(),b->deg()); i++){
        this->set_coeff(i,this->get_coeff(i) + b->get_coeff(i));
      }
    }
    void CPUMulAdd(Polynomial *b,ZZ M){
      // Forces the addition to be executed by CPU
      // This method supposes that there is no need to apply CRT/ICRT on operands

      // #pragma omp parallel for
      for( unsigned int i = 0; i <= std::max(this->deg(),b->deg()); i++){
        this->set_coeff(i,NTL::AddMod(this->get_coeff(i),b->get_coeff(i),M));
      }
    }

    void CPUSubtraction(Polynomial *b){
      // Forces the subtraction to be executed by CPU
      // This method supposes that there is no need to apply CRT/ICRT on operands
      for( int i = 0; i <= b->deg(); i++)
        this->set_coeff(i,this->get_coeff(i) - b->get_coeff(i));
    }

    void normalize(){
      if(!this->get_host_updated()){
        this->icrt();
      }

      // Remove last 0-coefficients
      while(this->deg() >= 0 &&
            this->get_coeff(this->deg()) == ZZ(0))
        this->coefs.pop_back();

    }
    // gets and sets
    ZZ get_mod(){
      // Returns a copy of mod
      return this->mod;
    }
    void set_mod(ZZ value){
      this->mod = value;
      #ifdef VERBOSE
        std::cout << "Polynomial mod set to " << this->mod << std::endl;
      #endif
    }
    Polynomial get_phi();

    void set_phi(Polynomial p){
      this->phi = &p;
      #ifdef VERBOSE
        std::cout << "Polynomial phi set to " << this->phi << std::endl;
      #endif
      Polynomial::phi_set = true;
    }


    ZZ get_coeff(const int index){

      if(!this->get_host_updated()){
        this->icrt();
      }

      // Returns a copy of coefficient at this index
      if(index > this->deg())
        return conv<ZZ>(0);
      else
        return this->coefs.at(index);
    }
    void set_coeff(int index,ZZ value){

      if(!this->get_host_updated()){
        this->icrt();
      }

      if((unsigned int)(index) >= this->coefs.size()){
        #ifdef VERBOSE
          std::cout << "Resizing this->coefs from "<< this->coefs.size() << " to " << index+1 << std::endl;
        #endif

        // this->coefs.resize((index == 0? 1024:this->coefs.size()+index + 1024));
        this->coefs.resize(index+1);
      }
      this->coefs[index] = value;
      this->expected_degree = this->deg();
      #ifdef DEBUG
        std::cout << "Polynomial coeff " << index << " set to " << this->coefs[index] << std::endl;
      #endif

        this->set_device_updated(false);
        this->set_host_updated(true);
    }
    void set_coeff(int index,int value){

      this->set_coeff(index,ZZ(value));
    }
    std::vector<ZZ> get_coeffs(){

      if(!this->get_host_updated()){
        this->icrt();
      }

      // Returns a copy of all coefficients
      std::vector<ZZ> coefs_copy(this->coefs);
      return coefs_copy;
    }
    void set_coeffs(std::vector<cuyasheint_t> values){

      // Replaces all coefficients
      this->coefs.resize(values.size());
      for(std::vector<cuyasheint_t>::iterator iter = values.begin();iter != values.end();iter++){
        this->coefs[iter-values.begin()] = conv<ZZ>(*iter);
      }
      this->expected_degree = this->deg();
      this->set_device_updated(false);
      this->set_host_updated(true);
    }
    void set_coeffs(std::vector<ZZ> values){

      // Replaces all coefficients
      this->coefs = values;
      this->expected_degree = this->deg();

      this->set_device_updated(false);
      this->set_host_updated(true);
    }
    void set_coeffs(){

      // Replaces all coefficients
      this->coefs.clear();
      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      this->expected_degree = this->coefs.size()-1;
      
      #ifdef DEBUGVERBOSE
        std::cout << "Polynomial coeff cleaned and resized to " << this->coefs.size() << std::endl;
      #endif

      this->set_device_updated(false);
      this->set_host_updated(true);
    }
     void set_coeffs(int size){
      // Prepare this polynomial to receive size coefficientes
      this->coefs.clear();
      this->coefs.resize(size);
      this->expected_degree = this->coefs.size()-1;
      
      #ifdef VERBOSE
        std::cout << "Polynomial coeff cleaned and resized to " << this->coefs.size() << std::endl;
      #endif

      this->set_device_updated(false);
      this->set_host_updated(true);
    }
    std::vector<std::vector<cuyasheint_t> > get_crt_residues(){
      std::vector<std::vector<cuyasheint_t> > crt_residues_copy(this->polyCRT);
      return crt_residues_copy;
    }
    cuyasheint_t* get_device_crt_residues(){
      // Returns the address of crt residues at device memory
      // if(this->d_polyCRT == NULL){
        // cudaError_t result = cudaMalloc((void**)&this->d_polyCRT,std::max(this->CRTSPACING,1)*(Polynomial::CRTPrimes.size())*sizeof(cuyasheint_t));
        // assert(result == cudaSuccess);
      // }
      return this->d_polyCRT;
    }
    void set_device_crt_residues(cuyasheint_t *residues){
      this->d_polyCRT = residues;
    }

    void crt();
    void icrt();
    int get_crt_spacing(){
      return this->CRTSPACING;
    }
    static void gen_crt_primes(ZZ q,cuyasheint_t degree){
        // We will use 63bit primes to fit cuyasheint_t data type (64 bits raises "GenPrime: length too large")
        ZZ M = ZZ(1);
        std::vector<cuyasheint_t> P;

        int primes_size = CRTPRIMESIZE;
        std::cout << "Primes size: " << primes_size << std::endl;
        cuyasheint_t n;

        while( (M < (2*degree)*q*q*q) ){
            n = NTL::GenPrime_long(primes_size);
            if( std::find(P.begin(), P.end(), n) == P.end()){
              // Does not contains
              P.push_back(n);
              M *=(n);
            }
        }

        Polynomial::CRTProduct = M;
        Polynomial::CRTPrimes = P;

        std::cout << P.size() << " primes generated." << std::endl;
        #ifdef DEBUG
        std::cout << "Primes set - M:" << Polynomial::CRTProduct << std::endl;
        // std::cout << "Primes: "<< Polynomial::CRTPrimes << std::endl;
        #endif
    }

    void update_device_data(unsigned int usable_ratio=1);
    void set_device_updated(bool b){
      this->DEVICE_IS_UPDATE = b;
      if(!b)
        this->set_crt_computed(false);
    }
    bool get_device_updated(){
      return this->DEVICE_IS_UPDATE;
    }
    void update_host_data();
    void set_host_updated(bool b){
      this->HOST_IS_UPDATED = b;
      if(!b)
        this->set_icrt_computed(false);
    }
    bool get_host_updated(){
      return this->HOST_IS_UPDATED;
    }
    void set_crt_computed(bool b){
      this->CRT_COMPUTED = b;
    }
    bool get_crt_computed(){
      return this->CRT_COMPUTED;
    }
    void set_icrt_computed(bool b){
      this->ICRT_COMPUTED = b;
    }
    bool get_icrt_computed(){
      return this->ICRT_COMPUTED;
    }

    int deg(){
      if(!this->get_host_updated())
        return this->expected_degree;        
      else
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
        cuyasheint_t * d_pointer = CUDAFunctions::callRealignCRTResidues(this->stream, this->CRTSPACING,new_spacing,this->get_device_crt_residues(),this->deg()+1,Polynomial::CRTPrimes.size());
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
    bool is_zero(){
      this->normalize();
      if(this->deg() == -1)
        return true;
      else
        return false;
    }
    bool is_one(){
      this->normalize();
      if(this->deg() == 0 && this->get_coeff(0) == 1)
        return true;
      else
        return false;
    }
    static void DivRem(Polynomial a,Polynomial b,Polynomial &quot,Polynomial &rem);
    void reduce();
    static Polynomial InvMod(Polynomial a,Polynomial b){
      // To-do
      // throw "Polynomial InvMod not implemented!";
      #warning "Polynomial InvMod not implemented!"

      //
      ZZ_pEX a_ntl;
      for(int i = 0; i <= a.deg();i++)
        NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
      ZZ_pEX b_ntl;
      for(int i = 0; i <= b.deg();i++)
        NTL::SetCoeff(b_ntl,i,conv<ZZ_p>(b.get_coeff(i)));

      ZZ_pEX inv_a_ntl =  NTL::InvMod(a_ntl, b_ntl);

      Polynomial result;
      for(int i = 0; i <= b.deg();i++){
        ZZ ntl_value;
        if( NTL::IsZero(NTL::coeff(inv_a_ntl,i)) )
        // Without this, NTL raises an exception when we call rep()
          ntl_value = 0L;
        else
          ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(inv_a_ntl,i))[0]);

        result.set_coeff(i,ntl_value);
      }

      return result;
    }
    // static void XGCD(Polynomial &d, Polynomial &s,Polynomial &t, Polynomial &a,  Polynomial &b);
    static void BuildNthCyclotomic(Polynomial *phi, unsigned int n);
    static void random(Polynomial *a,int degree){
      a->set_coeffs(degree+1);

      if(a->get_mod() > 0)
        for(int i = 0;i <= degree;i++)
          a->set_coeff(i,ZZ(rand())%a->get_mod());
      else
        for(int i = 0;i <= degree;i++)
          a->set_coeff(i,ZZ(rand()) % a->global_mod);
    }
  private:
    // Attributes
    cudaStream_t stream;
    int expected_degree; // This variable stores the expected degree for this polinomial
    std::vector<ZZ> coefs;
    std::vector<std::vector<cuyasheint_t> > polyCRT; // Must be initialized by crt()
    cuyasheint_t *d_polyCRT; // Must be initialized on CRTSPACING definition and updated by crt(), if needed
    ZZ mod;
    Polynomial *phi;

    bool ON_COPY;
    bool HOST_IS_UPDATED;
    bool DEVICE_IS_UPDATE;
    bool CRT_COMPUTED;
    bool ICRT_COMPUTED;
    //Functions and methods

};
#endif
