/**
 * cuYASHE
 * Copyright (C) 2015-2016 cuYASHE Authors
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H
#include <stdio.h>
#include <map>
#include <vector>
#include <sstream>
#include <NTL/ZZ.h>
#include <NTL/ZZ_pEX.h>
#include <NTL/ZZ_pXFactoring.h>
#include <algorithm>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include "cuda_bn.h"
#include "settings.h"
#include "common.h"

NTL_CLIENT

cuyasheint_t polynomial_get_cycles();
void get_words(bn_t* b,ZZ a);
void get_words_host(bn_t *b,ZZ a);
bn_t get_reciprocal(ZZ q);
bn_t get_reciprocal(bn_t q);
void compute_reciprocal(ZZ q);
extern std::map<ZZ, std::pair<cuyasheint_t*,int>> reciprocals;


// template Polynomial common_addition<Polynomial>(Polynomial *a,Polynomial *b);

class Polynomial{
  public:
    // Attributes
    int CRTSPACING =-1;// Stores the distance between the zero-coeff of two consecutive residues in d_polyCRT
    static ZZ CRTProduct;
    static std::vector<cuyasheint_t> CRTPrimes;
    static std::vector<ZZ> CRTMpi;
    static std::vector<cuyasheint_t> CRTInvMpi;
    static Polynomial *global_phi;
    static ZZ global_mod;
    static bool phi_set;

    // Constructors
    Polynomial(){
      #ifdef VERBOSE
      std::cout << "Building a polynomial" << std::endl;
      #endif
      cudaStreamCreate(&this->stream);
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(false);
      this->set_transf_computed(false);
      this->set_itransf_computed(false);
      if(Polynomial::global_mod > 0)
        // If a global mod is defined, use it
        this->mod = Polynomial::global_mod; // Doesn't copy. Uses the reference.
      
      if(Polynomial::global_phi){
        // If a global phi is defined, use it
        this->phi = Polynomial::global_phi; // Doesn't copy. Uses the reference.

        // If the irreductible polynomial have degree N, this polynomial's degree will be limited to N-1
        // if(Polynomial::global_phi->deg() >= 0)
          // this->update_crt_spacing(Polynomial::global_phi->deg());
      }
      

      if(Polynomial::phi_set)
        this->coefs.resize(this->get_phi().deg()+1);
      

      if(this->mod != 0)
        compute_reciprocal(this->mod);
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Polynomial(ZZ p){
      #ifdef VERBOSE
      std::cout << "Building a polynomial" << std::endl;
      #endif
      cudaStreamCreate(&this->stream);
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(false);
      this->set_transf_computed(false);
      this->set_itransf_computed(false);
      this->mod = ZZ(p);// Copy

      if(Polynomial::global_phi){
      //   // If a global phi is defined, use it
        this->phi = Polynomial::global_phi; // Doesn't copy. Uses the reference.
      }
      // this->update_crt_spacing(Polynomial::global_phi->deg());

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      
      if(this->mod != 0)
        compute_reciprocal(this->mod);
      
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod << " but no phi."<< std::endl;
      #endif
    }
    Polynomial(ZZ p,Polynomial P){
      #ifdef VERBOSE
      std::cout << "Building a polynomial" << std::endl;
      #endif
      cudaStreamCreate(&this->stream);
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(false);
      this->set_transf_computed(false);
      this->set_itransf_computed(false);
      this->mod = ZZ(p);// Copy
      this->phi = &P;// Copy

      // CRT Spacing should store the expected number of coefficients
      // If the irreductible polynomial have degree N, this polynomial's degree will be limited to N-1
      // this->update_crt_spacing(this->phi->deg()+1);

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      
      if(this->mod != 0)
        compute_reciprocal(this->mod);
      
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this-> phi << std::endl;
      #endif
    }
    Polynomial(int spacing){
      #ifdef VERBOSE
      std::cout << "Building a polynomial" << std::endl;
      #endif
      cudaStreamCreate(&this->stream);
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(false);
      this->set_transf_computed(false);
      this->set_itransf_computed(false);
      if(Polynomial::global_mod > 0){
        // If a global mod is defined, use it
        this->mod = Polynomial::global_mod; // Doesn't copy. Uses the reference.
      }
      if(Polynomial::global_phi){
      //   // If a global phi is defined, use it
        this->phi = Polynomial::global_phi; // Doesn't copy. Uses the reference.
      }
      // CRT Spacing set to spacing
      this->update_crt_spacing(spacing);
      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }

      
      if(this->mod != 0)
        compute_reciprocal(this->mod);
      
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Polynomial(ZZ p,Polynomial P,int spacing){
      #ifdef VERBOSE
      std::cout << "Building a polynomial" << std::endl;
      #endif
      cudaStreamCreate(&this->stream);
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(false);
      this->set_transf_computed(false);
      this->set_itransf_computed(false);
      this->mod = ZZ(p);// Copy
      this->phi = &P;// Copy

      // std::cout << this->get_phi().to_string() << std::endl;

      // CRT Spacing set to spacing
      this->update_crt_spacing(spacing);

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      
      if(this->mod != 0)
        compute_reciprocal(this->mod);
      
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this->phi << std::endl;
      #endif
    }
    Polynomial(ZZ p,int spacing){
      #ifdef VERBOSE
      std::cout << "Building a polynomial" << std::endl;
      #endif
      cudaStreamCreate(&this->stream);
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(false);
      this->set_transf_computed(false);
      this->set_itransf_computed(false);
      this->mod = ZZ(p);// Copy

      // CRT Spacing set to spacing
      this->update_crt_spacing(spacing);

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      
      if(this->mod != 0)
        compute_reciprocal(this->mod);
      
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod << std::endl;
      #endif
    }
    Polynomial(Polynomial *b){
      /////////////////////////////
      // Doesn't duplicate data //
      /////////////////////////////
      ///
      this->stream = b->get_stream();

      this->CRTSPACING = b->get_crt_spacing();
      this->d_bn_coefs = b->d_bn_coefs;
      this->h_bn_coefs = b->h_bn_coefs;
      this->d_polyCRT = b->d_polyCRT;
      #ifdef CUFFTMUL_TRANSFORM
      this->d_polyTransf = b->d_polyTransf;
      #endif

      this->set_crt_computed(b->get_crt_computed());
      this->set_icrt_computed(b->get_icrt_computed());
      this->set_transf_computed(b->get_transf_computed());
      this->set_itransf_computed(b->get_itransf_computed());
      this->set_host_updated(b->get_host_updated());

      if(b->get_host_updated())
        set_coeffs(b->get_coeffs());

    }
    void copy(Polynomial b){
      ///////////////////////////
      // Duplicates everything //
      ///////////////////////////
      #ifdef VERBOSE
      // cuyasheint_t start,stop;
      // start = polynomial_get_cycles();
      #endif 
      cudaDeviceSynchronize();

      this->update_crt_spacing(b.get_crt_spacing());
      if(b.get_icrt_computed()){
        // #warning this is wrong!
        cudaError_t result = cudaMemcpyAsync(this->d_bn_coefs,b.d_bn_coefs,b.get_crt_spacing()*sizeof(bn_t),cudaMemcpyDeviceToDevice,b.get_stream());
        assert(result == cudaSuccess);
        assert(CRTSPACING > 0);
        h_bn_coefs = (bn_t*)malloc(CRTSPACING*sizeof(bn_t));
      }

      this->set_crt_computed(b.get_crt_computed());
      this->set_icrt_computed(b.get_icrt_computed());
      this->set_transf_computed(b.get_transf_computed());
      this->set_itransf_computed(b.get_itransf_computed());
      this->set_host_updated(b.get_host_updated());

      if(b.get_crt_computed()){
        cudaError_t result = cudaMemcpyAsync(this->d_polyCRT,b.d_polyCRT,b.get_crt_spacing()*Polynomial::CRTPrimes.size()*sizeof(cuyasheint_t),cudaMemcpyDeviceToDevice,b.get_stream());
        assert(result == cudaSuccess);
      }
      if(b.get_transf_computed()){
        cudaError_t result = cudaMemcpyAsync(this->d_polyTransf,b.d_polyTransf,b.get_crt_spacing()*Polynomial::CRTPrimes.size()*sizeof(cuyasheint_t),cudaMemcpyDeviceToDevice,b.get_stream());
        assert(result == cudaSuccess);
      }

      if(b.get_host_updated())
        set_coeffs(b.get_coeffs());
      
      #ifdef VERBOSE
      // stop = polynomial_get_cycles();
      // std::cout << (stop-start) << " cycles to copy" << std::endl;
      #endif
    }

    ///////////////////////////
    // Functions and methods //
    ///////////////////////////

    void copy_device_crt_residues(Polynomial &b);

    std::string to_string(){
      #ifdef VERBOSE
      std::cout << "Will generate a string" << std::endl;
      #endif
      cudaError_t result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);

      if(!get_host_updated())
        update_host_data();
      
      stringstream ss;
      for(int i = 0; i <=  this->deg();i++)
        ss << this->get_coeff(i) << ", ";
      return ss.str();;
    }
    // Operators
    Polynomial operator=(Polynomial b){//Copy

      /////////////////////////////
      // Doesn't duplicate data //
      /////////////////////////////
      ///
      this->stream = b.get_stream();

      this->CRTSPACING = b.get_crt_spacing();
      this->d_bn_coefs = b.d_bn_coefs;
      this->h_bn_coefs = b.h_bn_coefs;
      this->d_polyCRT = b.d_polyCRT;
      this->d_polyTransf = b.d_polyTransf;

      this->set_crt_computed(b.get_crt_computed());
      this->set_icrt_computed(b.get_icrt_computed());
      this->set_transf_computed(b.get_transf_computed());
      this->set_itransf_computed(b.get_itransf_computed());
      this->set_host_updated(b.get_host_updated());

      if(b.get_host_updated())
        set_coeffs(b.get_coeffs());

        #ifdef VERBOSE
          std::cout << "Polynomial copied. " << std::endl;
        #endif
        return *this;
    }
        // Polynomial operator=(Polynomial b){//Copy
    //     this->stream = b.get_stream();

    //     this->CRTSPACING = b.get_crt_spacing();
    //     this->d_bn_coefs = b.d_bn_coefs;
    //     this->h_bn_coefs = b.h_bn_coefs;
    //     this->d_polyCRT = b.d_polyCRT;

    //     this->set_crt_computed(b.get_crt_computed());
    //     this->set_icrt_computed(b.get_icrt_computed());
    //     this->set_host_updated(b.get_host_updated());

    //     if(b.get_host_updated())
    //       set_coeffs(b.get_coeffs());
    //     return *this;
    // }
    Polynomial operator+(Polynomial &b);
    Polynomial operator+=(Polynomial &b);
    Polynomial operator-(Polynomial &b){
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
      // #pragma omp parallel sections
      {
          // #pragma omp section
          {

              if(!this->get_crt_computed())
                this->update_device_data();
          }
          // #pragma omp section
          {
              if(!b.get_crt_computed())
                  b.update_device_data();
          }
      }

      Polynomial *c;
      c = new Polynomial(this->get_mod(),this->get_phi(),this->get_crt_spacing());
      CUDAFunctions::callPolynomialAddSub(c->get_device_crt_residues(),
                                          this->get_device_crt_residues(),
                                          b.get_device_crt_residues(),
                                          (int)(this->CRTSPACING*Polynomial::CRTPrimes.size()),
                                          SUB,
                                          this->stream);

      c->set_host_updated(false);
      c->set_icrt_computed(false);
      c->set_crt_computed(true);
      // cudaDeviceSynchronize();
      return c;
    }
    Polynomial operator-=(Polynomial &b){
      // Check align
      if(this->CRTSPACING != b.CRTSPACING){
        int new_spacing = std::max(this->CRTSPACING,b.CRTSPACING);
        this->update_crt_spacing(new_spacing);
        b.update_crt_spacing(new_spacing);
      }

      #ifdef VERBOSE
      std::cout << "Sub:" << std::endl;
      #endif

      // Apply CRT and copy data to global memory, if needed
      // #pragma omp parallel sections
      {
          // #pragma omp section
          {

              if(!this->get_crt_computed())
                this->update_device_data();
          }
          // #pragma omp section
          {
              if(!b.get_crt_computed())
                  b.update_device_data();
          }
      }

      CUDAFunctions::callPolynomialAddSub(this->get_device_crt_residues(),
                                          this->get_device_crt_residues(),
                                          b.get_device_crt_residues(),
                                          (int)(this->CRTSPACING*Polynomial::CRTPrimes.size()),
                                          SUB,
                                          this->stream);

      this->set_host_updated(false);
      this->set_icrt_computed(false);
      this->set_crt_computed(true);
      // cudaDeviceSynchronize();
      return this;
    }
    Polynomial operator*(Polynomial &b);
    Polynomial operator*=(Polynomial &b);
    Polynomial operator/(Polynomial &b){
      Polynomial *quot = new Polynomial();
      Polynomial *rem = new Polynomial();

      // #pragma omp parallel sections
      // {
      //     #pragma omp section
      //     {
      //       if(!this->get_host_updated())
      //         this->update_host_data();
      //     }
      //     #pragma omp section
      //     {
      //       if(!b.get_host_updated())
      //         b.update_host_data();
      //     }
      // }

      Polynomial::DivRem((*this),b,(*quot), (*rem));
      return quot;
    }
    Polynomial operator/=(Polynomial &b){
      this->copy( ((*this)/b));
      return this;
    }
    Polynomial operator%(Polynomial &b){
      Polynomial *quot = new Polynomial();
      Polynomial *rem = new Polynomial();

      // #pragma omp parallel sections
      // {
      //     #pragma omp section
      //     {
      //       if(!this->get_host_updated())
      //         this->update_host_data();
      //     }
      //     #pragma omp section
      //     {
      //       if(!b.get_host_updated())
      //         b.update_host_data();
      //     }
      // }

      Polynomial::DivRem((*this),b,(*quot), (*rem));
      // rem.update_host_data();
      // std::cout << rem.to_string() << std::endl;
      return rem;
    }
    Polynomial operator%=(Polynomial &b){
      this->copy( ((*this)%b));
      return this;
    }
    // Polynomial operator+(ZZ b){
    //   if(this->get_crt_computed() || this->get_icrt_computed()){
    //     return Integer(b)+(*this);
    //   }else{
    //     Polynomial p(*this);
    //     p.set_coeff(0,p.get_coeff(0)+b);
    //     p.set_crt_computed(false);
    //     return p;
    //   }
    // }
    // Polynomial operator+=(ZZ b){
    //   Integer B(b);

    //   if(this->get_icrt_computed()){
    //     assert(this->get_crt_computed());
    //     assert(this->d_bn_coefs);
    //     CUDAFunctions::callPolynomialOPDigit( ADD,
    //                                         this->get_stream(),
    //                                         this->d_bn_coefs,
    //                                         B.get_digits(),
    //                                         this->deg()+1);
    //     this->set_crt_computed(false);
    //     this->set_host_updated(false);
    //   }else if(this->get_crt_computed()){
    //     CUDAFunctions::callPolynomialOPInteger( ADD,
    //                                             this->get_stream(),
    //                                             this->get_device_crt_residues(),
    //                                             B.get_device_crt_residues(),
    //                                             this->get_crt_spacing(),
    //                                             Polynomial::CRTPrimes.size());   
    //     this->set_host_updated(false);
    //   }else{
    //     assert(this->get_host_updated());  
    //     this->set_coeff(0,this->get_coeff(0)-b);
    //   }
    //   return *this;
    // }
    // Polynomial operator-(ZZ b){
    //  if(this->get_crt_computed() || this->get_icrt_computed()){
    //     return Integer(b)-(*this);
    //   }else{
    //     Polynomial p(*this);
    //     p.set_coeff(0,p.get_coeff(0)-b);
    //     p.set_crt_computed(false);
    //     return p;
    //   }
    // }
    // Polynomial operator-=(ZZ b){
    //   Integer B(b);

    //   if(this->get_icrt_computed()){
    //     assert(this->d_bn_coefs);
    //     CUDAFunctions::callPolynomialOPDigit( SUB,
    //                                         this->get_stream(),
    //                                         this->d_bn_coefs,
    //                                         B.get_digits(),
    //                                         this->deg()+1);
    //     this->set_crt_computed(false);
    //     this->set_host_updated(false);
    //   }else if(this->get_crt_computed()){
    //     assert(this->d_polyCRT);
    //     CUDAFunctions::callPolynomialOPInteger( SUB,
    //                                             this->get_stream(),
    //                                             this->get_device_crt_residues(),
    //                                             B.get_device_crt_residues(),
    //                                             this->get_crt_spacing(),
    //                                             Polynomial::CRTPrimes.size());   
    //     this->set_host_updated(false);
    //   }else{
    //     assert(this->get_host_updated());  
    //     this->set_coeff(0,this->get_coeff(0)-b);
    //   }
    //   return *this;
    // }
    Polynomial operator*(ZZ b){
      Polynomial p(*this);

      if(p.get_icrt_computed() || p.get_crt_computed() || p.get_transf_computed()){
        p.icrt();

        assert(p.d_bn_coefs);
        if(p.deg()+1 == 0)
          return p;
        
        bn_t B;
        get_words(&B,b);
        CUDAFunctions::callPolynomialOPDigit( MUL,
                                              p.get_stream(),
                                              p.d_bn_coefs,
                                              B,
                                              p.deg()+1);
        p.set_crt_computed(false);
        p.set_host_updated(false);
      }else{
        assert(p.get_host_updated()); 
        for(int i = 0; i <= p.deg();i++)
          p.set_coeff(i,p.get_coeff(i)*b);
      }
      return p;
    }
    Polynomial operator*=(ZZ b){
      
      if(get_icrt_computed() || get_crt_computed() || get_transf_computed()){
        icrt();

        assert(this->d_bn_coefs);
        if(this->deg()+1 == 0)
          return *this;
        
        bn_t B;
        get_words(&B,b);
        CUDAFunctions::callPolynomialOPDigit( MUL,
                                            this->get_stream(),
                                            this->d_bn_coefs,
                                            B,
                                            this->deg()+1);
        this->set_crt_computed(false);
        this->set_host_updated(false);
      }else{
        assert(this->get_host_updated()); 
        for(int i = 0; i <= this->deg();i++)
          this->set_coeff(i,this->get_coeff(i)*b);
      }
      return *this;
    }
    Polynomial operator%(ZZ b){

        Polynomial p(*this);
        p %= b;
        return p;
    }
    Polynomial operator%=(ZZ b){
      if(get_icrt_computed() || get_crt_computed()){
        icrt();
        modn(b);
        return *this;                
      }

      // #pragma omp parallel for
      for(int i = 0; i <= this->deg();i++)
        this->set_coeff(i,this->get_coeff(i)%b);
      
      this->set_crt_computed(false);

      return *this;
    }
    // Polynomial operator/(ZZ b){
    //   Polynomial p(*this);
    //   if(!p.get_host_updated()){
    //     // #warning "Polynomial division on device not implemented!";
    //     p.update_host_data();
    //   }

    //   // #pragma omp parallel for
    //   for(int i = 0; i <= p.deg();i++)
    //     p.set_coeff(i,p.get_coeff(i)/b);
    //   p.set_crt_computed(false);

    //   return p;
    // }
    // Polynomial operator/=(ZZ b){
    //   this->copy(((*this)/b));
    //   return *this;
    // }
    // // Polynomial operator%(bn_t b){
    // //   Polynomial p(*this);

    // // }
    // Polynomial operator%=(bn_t b){
    //   if(!this->get_crt_computed())
    //     this->crt();      
    //   modn(b);
    //   return *this;
    // }
    Polynomial operator+(cuyasheint_t b){
      Polynomial *p = new Polynomial(get_mod(), get_phi(), get_crt_spacing());

      if(!get_host_updated()){
        #ifdef NTTMUL_TRANSFORM
        set_device_transf_residues( CUDAFunctions::callPolynomialOPInteger(
                                                      ADD,
                                                      get_stream(),
                                                      get_device_crt_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size())
          );
        #else
        set_device_transf_residues( CUDAFunctions::callPolynomialcuFFTOPInteger(
                                                      ADD,
                                                      get_stream(),
                                                      get_device_transf_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size())
          );
        #endif
      }else{
        for(int i = 0; i <= deg(); i++)
          p->set_coeff(i,get_coeff(i)+b);
      }

      return p;
    }
    Polynomial operator+=(cuyasheint_t b){
      if(!get_host_updated()){
        #ifdef NTTMUL_TRANSFORM
        CUDAFunctions::callPolynomialOPIntegerInplace(
                                                      ADD,
                                                      get_stream(),
                                                      get_device_crt_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size());
        #else
        CUDAFunctions::callPolynomialcuFFTOPIntegerInplace(
                                                      ADD,
                                                      get_stream(),
                                                      get_device_transf_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size());
        #endif
      }else{
        for(int i = 0; i <= deg(); i++)
          set_coeff(i,get_coeff(i)+b);
      }

      return *this;
    }
    Polynomial operator-(cuyasheint_t b){
      Polynomial *p = new Polynomial(get_mod(), get_phi(), get_crt_spacing());

      if(!get_host_updated()){
        #ifdef NTTMUL_TRANSFORM
        set_device_transf_residues( CUDAFunctions::callPolynomialOPInteger(
                                                      SUB,
                                                      get_stream(),
                                                      get_device_crt_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size())
          );
        #else
        set_device_transf_residues( CUDAFunctions::callPolynomialcuFFTOPInteger(
                                                      SUB,
                                                      get_stream(),
                                                      get_device_transf_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size())
          );
        #endif
      }else{
        for(int i = 0; i <= deg(); i++)
          p->set_coeff(i,get_coeff(i)-b);
      }

      return p;
    }
    Polynomial operator-=(cuyasheint_t b){
      if(!get_host_updated()){
        #ifdef NTTMUL_TRANSFORM
        CUDAFunctions::callPolynomialOPIntegerInplace(
                                                      SUB,
                                                      get_stream(),
                                                      get_device_crt_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size());
        #else
        CUDAFunctions::callPolynomialcuFFTOPIntegerInplace(
                                                      SUB,
                                                      get_stream(),
                                                      get_device_transf_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size());
        #endif
      }else{
        for(int i = 0; i <= deg(); i++)
          set_coeff(i,get_coeff(i)-b);
      }

      return *this;
    }
    Polynomial operator*(cuyasheint_t b){
      Polynomial *p = new Polynomial(get_mod(), get_phi(), get_crt_spacing());

      if(!get_host_updated()){
        #ifdef NTTMUL_TRANSFORM
        set_device_transf_residues( CUDAFunctions::callPolynomialOPInteger(
                                                      MUL,
                                                      get_stream(),
                                                      get_device_crt_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size())
          );
        #else
        set_device_transf_residues( CUDAFunctions::callPolynomialcuFFTOPInteger(
                                                      MUL,
                                                      get_stream(),
                                                      get_device_transf_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size())
          );
        #endif
      }else{
        for(int i = 0; i <= deg(); i++)
          p->set_coeff(i,get_coeff(i)*b);
      }

      return p;
    }
    Polynomial operator*=(cuyasheint_t b){
      if(!get_host_updated()){
        #ifdef NTTMUL_TRANSFORM
        CUDAFunctions::callPolynomialOPIntegerInplace(
                                                      MUL,
                                                      get_stream(),
                                                      get_device_crt_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size());
        #else
        CUDAFunctions::callPolynomialcuFFTOPIntegerInplace(
                                                      MUL,
                                                      get_stream(),
                                                      get_device_transf_residues(),
                                                      b,
                                                      get_crt_spacing(),
                                                      CRTPrimes.size());
        #endif
      }else{
        for(int i = 0; i <= deg(); i++)
          set_coeff(i,get_coeff(i)*b);
      }

      return *this;
    }
    Polynomial operator%(cuyasheint_t b){
      return (*this)%ZZ(b);
    }
    Polynomial operator%=(cuyasheint_t b){
      *this %= ZZ(b);
      return *this;
    }
    // Polynomial operator/(cuyasheint_t b){
    //   return (*this)/ZZ(b);
    // }
    // Polynomial operator/=(cuyasheint_t b){
    //   *this /= ZZ(b);
    //   return *this;
    // }
    bool operator==(Polynomial b){
      if(!this->get_host_updated())
          this->update_host_data();
      
      if(!b.get_host_updated())
        b.update_host_data();
      
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
        this->update_host_data();

      // #pragma omp parallel for
      for(int i = 0; i <= this->deg();i++)
        this->set_coeff(i,NTL::MulMod(this->get_coeff(i),b,mod));
      this->set_crt_computed(false);
    }
    void CPUAddition(Polynomial *b){
      // Forces the addition to be executed by CPU
      // This method supposes that there is no need to apply CRT/ICRT on operands
      for( int i = 0; i <= std::max(this->deg(),b->deg()); i++){
        this->set_coeff(i,this->get_coeff(i) + b->get_coeff(i));
      }
    }
    void CPUMulAdd(Polynomial *b,ZZ M){
      // Forces the addition to be executed by CPU
      // This method supposes that there is no need to apply CRT/ICRT on operands

      // #pragma omp parallel for
      for( int i = 0; i <= std::max(this->deg(),b->deg()); i++){
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
      if(!this->get_host_updated())
        this->update_host_data();

      // Remove last 0-coefficients
      while(this->coefs.size() > 0 &&
            this->coefs.back() == 0)
        this->coefs.pop_back();
      int new_spacing = 2*pow(2,ceil(log2(deg())));
      // int new_spacing = 2*pow(2,ceil(log2(Polynomial::global_phi->deg())));
      this->update_crt_spacing(new_spacing);
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

      if(!this->get_host_updated())
        this->update_host_data();
      

      // Returns a copy of coefficient at this index
      if(index > this->deg())
        return conv<ZZ>(0);
      else
        return this->coefs.at(index);
    }
    void set_coeff(int index,ZZ value){

      // if(!this->get_host_updated()){
        // this->update_host_data();
      // }

      if((unsigned int)(index) >= this->coefs.size()){
        if(value == 0)
          return;

        #ifdef VERBOSE
          std::cout << "Resizing this->coefs from "<< this->coefs.size() << " to " << index+1 << std::endl;
        #endif

        // this->coefs.resize((index == 0? 1024:this->coefs.size()+index + 1024));
        this->coefs.resize(index+1);
      }
      this->coefs[index] = value;
      #ifdef DEBUG
        std::cout << "Polynomial coeff " << index << " set to " << this->coefs[index] << std::endl;
      #endif

        this->set_crt_computed(false);
        this->set_icrt_computed(false);
        this->set_host_updated(true);
        if(this->get_crt_spacing() < 2*(this->deg())+1)
          this->update_crt_spacing(2*(this->deg())+1);
    }
    void set_coeff(int index,int value){

      this->set_coeff(index,ZZ(value));
    }
    std::vector<ZZ> get_coeffs(){

      if(!this->get_host_updated()){
        this->update_host_data();
      }

      // Returns a copy of all coefficients
      std::vector<ZZ> coefs_copy(this->coefs);
      return coefs_copy;
    }
    void set_coeffs(std::vector<cuyasheint_t> values){

      // Replaces all coefficients
      this->coefs.clear();
      this->coefs.resize(values.size());
      for(std::vector<cuyasheint_t>::iterator iter = values.begin();iter != values.end();iter++)
        this->coefs[iter-values.begin()] = conv<ZZ>(*iter);
      
      this->set_crt_computed(false);
      this->set_icrt_computed(false);
      this->set_transf_computed(false);
      this->set_itransf_computed(false);
      this->set_host_updated(true);
      if(this->get_crt_spacing() < (2*(this->deg()+1)))
        this->update_crt_spacing(2*(this->deg()+1));  
    }
    void set_coeffs(std::vector<ZZ> values){

      // Replaces all coefficients
      this->coefs.clear();
      this->coefs.resize(values.size());
      for(std::vector<ZZ>::iterator iter = values.begin();iter != values.end();iter++)
        this->coefs[iter-values.begin()] = *iter;

      this->set_crt_computed(false);
      this->set_icrt_computed(false);
      this->set_transf_computed(false);
      this->set_itransf_computed(false);
      this->set_host_updated(true);
      if(this->get_crt_spacing() < (2*(this->deg()+1)))
        this->update_crt_spacing(2*(this->deg()+1));  
    }
    void set_coeffs(){

      // Replaces all coefficients
      this->coefs.clear();
      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      
      #ifdef DEBUGVERBOSE
        std::cout << "Polynomial coeff cleaned and resized to " << this->coefs.size() << std::endl;
      #endif

      // this->set_crt_computed(false);
        // this->set_icrt_computed(false);
      // this->set_host_updated(true);
      // if(this->get_crt_spacing() < (this->deg()+1))
        // this->set_crt_spacing(Polynomial::global_phi->deg());

    }
     void set_coeffs(int size){
      // Prepare this polynomial to receive size coefficients
      this->coefs.clear();
      this->coefs.resize(size);
      
      #ifdef VERBOSE
        std::cout << "Polynomial coeff cleaned and resized to " << this->coefs.size() << std::endl;
      #endif

      // this->set_crt_computed(false);
        // this->set_icrt_computed(false);
      // this->set_host_updated(true);      
      // this->update_crt_spacing(size);
    }
    cuyasheint_t* get_device_crt_residues(){
      // Returns the address of crt residues at device memory
      if(this->d_polyCRT == NULL){
        this->CRTSPACING = std::max(this->get_crt_spacing(),1);
        cudaError_t result = cudaMalloc((void**)&this->d_polyCRT,this->get_crt_spacing()*(Polynomial::CRTPrimes.size())*sizeof(cuyasheint_t));
        assert(result == cudaSuccess);
      }
      // this->update_crt_spacing(this->get_crt_spacing());
      return this->d_polyCRT;
    }
    #ifdef CUFFTMUL_TRANSFORM
    Complex* get_device_transf_residues(){
      // Returns the address of crt residues at device memory
      return this->d_polyTransf;
    }
    void set_device_transf_residues(Complex *residues){
      d_polyTransf = residues;
    }
    #endif
    void set_device_crt_residues(cuyasheint_t *residues){
      d_polyCRT = residues;
    }

    int get_crt_spacing(){
      // this->update_crt_spacing(2*Polynomial::global_phi->deg());

      return this->CRTSPACING;
    }

    static void gen_crt_primes(ZZ q,cuyasheint_t degree){
        ZZ M = ZZ(1);
        std::vector<cuyasheint_t> P;
        std::vector<ZZ> Mpi;
        std::vector<cuyasheint_t> InvMpi;

        cuyasheint_t n;

        // Get primes
        // std::cout << "Primes: " << std::endl;
        #ifdef CUFFTMUL_TRANSFORM
        int primes_size = CRTPRIMESIZE;
        #else
        unsigned int count = 0;
        #endif
        while( (M < (2*degree)*q*q) ){
            
            #ifdef CUFFTMUL_TRANSFORM
            n = NTL::GenPrime_long(primes_size);
            #else
            n = PRIMES_BUCKET[count];
            count++;
            #endif

            if( std::find(P.begin(), P.end(), n) == P.end()){
              // Does not contains
              // std::cout << n << std::endl;
              P.push_back(n);
              M *=(n);
            }
        }
        // std::cout << std::endl;
        // Compute M/pi and it's inverse
        for(unsigned int i = 0; i < P.size();i++){
          ZZ pi = to_ZZ(P[i]);
          Mpi.push_back(M/pi);
          InvMpi.push_back(conv<cuyasheint_t>(NTL::InvMod(Mpi[i]%pi,pi)));
        }

        #ifndef CUFFTMUL_TRANSFORM
        compute_reciprocal(M);
        #endif

        Polynomial::CRTProduct = M;
        Polynomial::CRTPrimes = P;
        Polynomial::CRTMpi = Mpi;
        Polynomial::CRTInvMpi = InvMpi;

        #ifdef VERBOSE
        std::cout << "Primes size: " << primes_size << std::endl;
        std::cout << "Primes set - M:" << Polynomial::CRTProduct << std::endl;
        std::cout << P.size() << " primes generated." << std::endl;
        #endif

        // Send primes to GPU
        CUDAFunctions::write_crt_primes();
    }



    static void gen_crt_primes(ZZ q,cuyasheint_t degree, int primes_size){
        // We will use 63bit primes to fit cuyasheint_t data type (64 bits raises "GenPrime: length too large")
        ZZ M = ZZ(1);
        std::vector<cuyasheint_t> P;
        std::vector<ZZ> Mpi;
        std::vector<cuyasheint_t> InvMpi;

        cuyasheint_t n;

        // Get primes
        // std::cout << "Primes: " << std::endl;
        // unsigned int count = 0;

        while( (M < (2*degree)*q*q) ){
            
            // #ifdef CUFFTMUL_TRANSFORM
            n = NTL::GenPrime_long(primes_size);
            // #else
            // n = PRIMES_BUCKET[count];
            // count++;
            // #endif

            if( std::find(P.begin(), P.end(), n) == P.end()){
              // Does not contains
              // std::cout << n << std::endl;
              P.push_back(n);
              M *=(n);
            }
        }
        // std::cout << std::endl;
        // Compute M/pi and it's inverse
        for(unsigned int i = 0; i < P.size();i++){
          ZZ pi = to_ZZ(P[i]);
          Mpi.push_back(M/pi);
          InvMpi.push_back(conv<cuyasheint_t>(NTL::InvMod(Mpi[i]%pi,pi)));
        }

        compute_reciprocal(M);

        Polynomial::CRTProduct = M;
        Polynomial::CRTPrimes = P;
        Polynomial::CRTMpi = Mpi;
        Polynomial::CRTInvMpi = InvMpi;

        #ifdef VERBOSE
        std::cout << "Primes size: " << primes_size << std::endl;
        std::cout << "Primes set - M:" << Polynomial::CRTProduct << std::endl;
        std::cout << P.size() << " primes generated." << std::endl;
        #endif

        // Send primes to GPU
        CUDAFunctions::write_crt_primes();
    }

    void update_device_data();
    void update_host_data();
    void set_host_updated(bool b){
      this->HOST_IS_UPDATED = b;
      if(b){        
        // #ifdef VERBOSE
        // std::cout << "Host data is updated" << std::endl;
        // #endif
      }else{        
        // this->set_coeffs();
        // #ifdef VERBOSE
        // std::cout << "Host data is NOT updated" << std::endl;
        // #endif
      }
    }
    bool get_host_updated(){

      bool b = this->HOST_IS_UPDATED;
      // if(b){        
      //   #ifdef VERBOSE
      //   std::cout << "Host data is updated" << std::endl;
      //   #endif
      // }else{        
      //   this->set_icrt_computed(false);
      //   #ifdef VERBOSE
      //   std::cout << "Host data is NOT updated" << std::endl;
      //   #endif
      // }

      return b;
    }
    void set_crt_computed(bool b){
      this->CRT_COMPUTED = b;
      if(b){        
        // #ifdef VERBOSE
        // std::cout << "CRT residues computed" << std::endl;
        // #endif
      }else{        
        // #ifdef VERBOSE
        // std::cout << "CRT residues NOT computed" << std::endl;
        // #endif
      }
    }
    bool get_crt_computed(){
      bool b = this->CRT_COMPUTED;

      // if(b){        
      //   #ifdef VERBOSE
      //   std::cout << "CRT residues computed" << std::endl;
      //   #endif
      // }else{        
      //   #ifdef VERBOSE
      //   std::cout << "CRT residues NOT computed" << std::endl;
      //   #endif
      // }

      return b;
    }
    void set_icrt_computed(bool b){
      this->ICRT_COMPUTED = b;

      if(b){        
        // #ifdef VERBOSE
        // std::cout << "ICRT residues computed" << std::endl;
        // #endif
      }else{        
        // #ifdef VERBOSE
        // std::cout << "ICRT residues NOT computed" << std::endl;
        // #endif
      }
    }
    bool get_transf_computed(){
      bool b = this->TRANSF_COMPUTED;
      return b;
    }
    void set_transf_computed(bool b){
      this->TRANSF_COMPUTED = b;
    }

    bool get_itransf_computed(){
      bool b = this->ITRANSF_COMPUTED;
      return b;
    }
    void set_itransf_computed(bool b){
      this->ITRANSF_COMPUTED = b;
    }
    bool get_icrt_computed(){
      bool b = this->ICRT_COMPUTED;

      // if(b){        
      //   #ifdef VERBOSE
      //   std::cout << "ICRT residues computed" << std::endl;
      //   #endif
      // }else{        
      //   #ifdef VERBOSE
      //   std::cout << "ICRT residues NOT computed" << std::endl;
      //   #endif
      // }

      return b;
    }
    void crt();
    void icrt();
    void transf();
    void itransf();
    void modn(ZZ n){
      cudaError_t result;
      
      // Compute ICRT, if needed
      icrt();

      // Get the words
      bn_t *h_m;
      h_m = (bn_t*)malloc(sizeof(bn_t));
      h_m->alloc = 0;
      get_words(h_m,n);
        
      bn_t u = get_reciprocal(n);

      // Apply the operation
      callCuModN( d_bn_coefs,
                  d_bn_coefs,
                  get_crt_spacing(),
                  h_m->dp,
                  h_m->used,
                  u.dp,
                  u.used,
                  get_stream());
      result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);

      // Set status
      set_host_updated(false);
      set_crt_computed(false);
      set_icrt_computed(true);

      // Release memory
      
      bn_free(h_m);
      free(h_m);
    }
  void modn(bn_t n){

      /**
       * n must be a hosts variable with a pointer to device memory
       */

      // cudaError_t result;

      bn_t u = get_reciprocal(n);

      callCuModN( d_bn_coefs,
                  d_bn_coefs,
                  get_crt_spacing(),
                  n.dp,
                  n.used,
                  u.dp,
                  u.used,
                  get_stream());
    }
    int deg(){
      if(!get_host_updated()){
        icrt();
        return callBNGetDeg(d_bn_coefs,get_crt_spacing());
      }else
        return coefs.size()-1;
    }
    ZZ lead_coeff(){
      if(this->deg() >= 0){
        return this->get_coeff(this->deg());
      }else{
        return ZZ(0);
      }
    }
    static bool check_special_rem_format(Polynomial *p){
      if(p->get_coeff(0) != 1){
        return false;
      }
      if(p->lead_coeff() != ZZ(1)){
        return false;
      }
      for(int i =1;i < p->deg();i++){
        if(p->get_coeff(i) != 0)
          return false;
      }
      return true;
    }
    void update_crt_spacing(const int spacing){
      
      #ifdef SPEEDCHECK
      std::cout << "update_crt_spacing - " << spacing << std::endl;
      #endif

      int new_spacing;
      if(spacing <= 0)
        new_spacing = pow(2,ceil(log2(get_phi().deg())));
      else
        new_spacing = pow(2,ceil(log2(spacing)));

      /**
       * Adjust CRTSPACING
       *
       * Realloc *_bn_coefs and d_poly_CRT either
       */
      if(this->get_crt_spacing() == new_spacing){
        // Data lies in GPU's global memory and has the correct alignment
        #ifdef SPEEDCHECK
        std::cout << "No need to update crt spacing." << std::endl;
        #endif
        return;
      }else if(!(get_crt_computed() || get_transf_computed())){
        update_host_data();
        
        #ifdef SPEEDCHECK
        std::cout << "Will alloc memory  to update crt spacing." << std::endl;
        #endif
        cudaError_t result;

        // Data isn't updated on GPU's global memory
        // Just set the spacing and update gpu
        this->CRTSPACING = new_spacing;

        /**
         * Update bn_coefs
         */
        // #warning memory leak here
       // try{
       //  if(d_bn_coefs){
       //    result = cudaFree(d_bn_coefs);
       //    d_bn_coefs = 0x0;
       //    if(result != cudaSuccess)
       //      throw  cudaGetErrorString(result);
       //  }
       // }catch(const char* s){
       //  std::cerr << "Exception on release of d_bn_coefs: " << s << std::endl;
       //  cudaGetLastError();// Reset
       // }

       // try{
       //  if(d_polyCRT){
       //    result = cudaFree(d_polyCRT);
       //    d_polyCRT = 0x0;
       //    if(result != cudaSuccess)
       //      throw  cudaGetErrorString(result);
       //  }
       // }catch(const char* s){
       //  std::cerr << "Exception on release of d_polyCRT: " << s << std::endl;
       //  cudaGetLastError();// Reset
       // } 
       // try{
       //  if(h_bn_coefs){
       //    free(h_bn_coefs);
       //    h_bn_coefs = 0x0;
       //  }
       // }catch(const char* s){
       //  std::cerr << "Exception on release of h_bn_coefs: " << s << std::endl;
       // }
        
        // Alloc memory
        cuyasheint_t *tmp;
        // result = cudaMalloc((void**)&tmp,new_spacing*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));
        // assert(result == cudaSuccess);
        result = cudaMalloc((void**)&d_bn_coefs,new_spacing*sizeof(bn_t));
        assert(result == cudaSuccess);
        /**
         * We use a single cudaMalloc call for tmp and d_polyCRT
         */
        // result = cudaMalloc((void**)&d_polyCRT,(new_spacing*(CRTPrimes.size())+new_spacing*STD_BNT_WORDS_ALLOC)*sizeof(cuyasheint_t));        
        // assert(result == cudaSuccess);
        #ifdef CUFFTMUL_TRANSFORM
        result = cudaMalloc((void**)&d_polyTransf,new_spacing*(CRTPrimes.size())*sizeof(Complex));        
        assert(result == cudaSuccess);
        #endif
        result = cudaMalloc((void**)&d_polyCRT,new_spacing*(CRTPrimes.size())*sizeof(cuyasheint_t));        
        if(result != cudaSuccess){
          size_t t,f;
      cudaMemGetInfo(&f, &t);
      cout << "Used memory: " << (t-f)/(1024*1024) << std::endl;
      cout << "Free memory: " << f/(1024*1024) << std::endl;

          std::cout << cudaGetErrorString(result) << " - " << new_spacing*(CRTPrimes.size())*sizeof(cuyasheint_t) << " bytes" << std::endl;
        }

        assert(result == cudaSuccess);
        result = cudaMalloc((void**)&tmp,new_spacing*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));        
        assert(result == cudaSuccess);
        // tmp = &d_polyCRT[new_spacing*(CRTPrimes.size())];

        h_bn_coefs = (bn_t*)malloc(new_spacing*sizeof(bn_t));

        result = cudaMemsetAsync(tmp,0,new_spacing*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t),get_stream());
        assert(result == cudaSuccess);

        for(int i = 0; i < new_spacing; i++,tmp += STD_BNT_WORDS_ALLOC){
          h_bn_coefs[i].alloc = STD_BNT_WORDS_ALLOC;
          h_bn_coefs[i].used = 0;
          h_bn_coefs[i].sign = BN_POS;
          h_bn_coefs[i].dp = tmp;
        }

        result = cudaMemcpyAsync(d_bn_coefs,h_bn_coefs,new_spacing*sizeof(bn_t),cudaMemcpyHostToDevice,get_stream());
        assert(result == cudaSuccess);

        /**
         * Update residues array
         */
        #ifdef CUFFTMUL_TRANSFORM
        result = cudaMemsetAsync(d_polyTransf,0,new_spacing*(CRTPrimes.size())*sizeof(Complex),get_stream());
        assert(result == cudaSuccess);
        #endif
        result = cudaMemsetAsync(d_polyCRT,0,new_spacing*(CRTPrimes.size())*sizeof(cuyasheint_t),get_stream());
        assert(result == cudaSuccess);

        if(get_icrt_computed())
          crt();

        return; 
      }else{
        #ifdef SPEEDCHECK
        std::cout << "Need a realign to update crt spacing." << std::endl;
        #endif
        cudaError_t result;
        /**
         * Update bn_coefs
         */
       try{
        if(d_bn_coefs){
          //result = cudaFree(d_bn_coefs);
          d_bn_coefs = 0x0;
          // if(result != cudaSuccess)
          //   throw  cudaGetErrorString(result);
        } 
       }catch(const char* s){
        std::cerr << "Exception on release of d_bn_coefs: " << s << std::endl;
        cudaGetLastError();// Reset
       } 
       try{
         if(h_bn_coefs){
            // free(h_bn_coefs);
            h_bn_coefs = 0x0;
         }
       }catch(const char* s){
        std::cerr << "Exception on release of h_bn_coefs: " << s << std::endl;
        cudaGetLastError();// Reset
       }

        cuyasheint_t *tmp;
        result = cudaMalloc((void**)&tmp,new_spacing*STD_BNT_WORDS_ALLOC*sizeof(cuyasheint_t));
        assert(result == cudaSuccess);
        result = cudaMalloc((void**)&d_bn_coefs,new_spacing*sizeof(bn_t));
        assert(result == cudaSuccess);
        h_bn_coefs = (bn_t*)malloc(new_spacing*sizeof(bn_t));

        for(int i = 0; i < new_spacing; i++,tmp += STD_BNT_WORDS_ALLOC){
          h_bn_coefs[i].alloc = STD_BNT_WORDS_ALLOC;
          h_bn_coefs[i].used = 0;
          h_bn_coefs[i].sign = BN_POS;
          h_bn_coefs[i].dp = tmp;
        }

        result = cudaMemcpyAsync(d_bn_coefs,h_bn_coefs,new_spacing*sizeof(bn_t),cudaMemcpyHostToDevice,get_stream());
        assert(result == cudaSuccess);

        /**
         * Update residues array
         */
        
        // GPU has the updated data, but with wrong spacing
        // If updated data lies in gpu's global memory, realign it

        const int old_spacing = this->get_crt_spacing();
        cuyasheint_t * d_pointer = CUDAFunctions::callRealignCRTResidues(this->stream,
                                                                        this->get_crt_spacing(),
                                                                        new_spacing,
                                                                        this->get_device_crt_residues(),
                                                                        old_spacing,
                                                                        Polynomial::CRTPrimes.size());
        if(d_pointer != NULL){
          this->set_device_crt_residues(d_pointer);
        }else{
          #ifdef VERBOSE
          std::cout << "Old spacing is equal new spacing." << std::endl;
          #endif
        }

        this->CRTSPACING = new_spacing;
        
        #ifdef VERBOSE
        std::cout << "crt spacing updated to " << this->get_crt_spacing() << std::endl;
        #endif
      }
    }
    void update_crt_spacing(){
      int new_spacing = this->deg()+1;
      if(this->get_crt_spacing() == new_spacing)
        return;
      else
        update_crt_spacing(new_spacing);
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
      ///////////////////////////////
      // This is a very ugly hack. //
      ///////////////////////////////
      a.update_host_data();
      b.update_host_data();

      //
      ZZ_pEX a_ntl;
      for(int i = 0; i <= a.deg();i++)
        NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
      ZZ_pEX b_ntl;
      for(int i = 0; i <= b.deg();i++)
        NTL::SetCoeff(b_ntl,i,conv<ZZ_p>(b.get_coeff(i)));

      ZZ_pEX inv_a_ntl =  NTL::InvMod(a_ntl, b_ntl);

      Polynomial result;
      result.set_coeffs(b.deg()+1);
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
    static void random(Polynomial *a,const int degree){
      a->set_coeffs((degree+1));

      if(a->get_mod() > 0)
        for(int i = 0;i <= degree;i++)
          a->set_coeff(i,ZZ(rand())%a->get_mod());
      else
        for(int i = 0;i <= degree;i++)
          a->set_coeff(i,ZZ(rand()) % a->global_mod);
      a->set_transf_computed(false);
      a->set_itransf_computed(false);
      a->set_crt_computed(false);
      a->set_icrt_computed(false);
      a->set_host_updated(true);
      a->normalize();
    }
    cudaStream_t get_stream(){
      // return this->stream;
      return (cudaStream_t)0;
    }
    void set_stream(){
      cudaStreamCreate(&this->stream);
    }
    static void operator delete(void *ptr);
    void release(){
      cudaError_t result = cudaDeviceSynchronize();
      assert(result == cudaSuccess);
      
      if(d_polyCRT){
        result = cudaFree(d_polyCRT);
        assert(result == cudaSuccess);
        d_polyCRT = 0x0;
      }
      if(d_bn_coefs){
        result = cudaFree(d_bn_coefs);
        assert(result == cudaSuccess);
        d_bn_coefs = 0x0;
      }
      if(h_bn_coefs){
        if(h_bn_coefs[0].dp){
          // All h_bn_coefs[*].dp are alloc in a single array
          // result = cudaFree(h_bn_coefs[0].dp);
          assert(result == cudaSuccess);
        }
        free(h_bn_coefs);
        h_bn_coefs = 0x0;
      }
    }
    bn_t* h_bn_coefs = 0x0;
    bn_t* d_bn_coefs = 0x0;
  private:
    // Attributes
    bool ON_COPY;
    bool HOST_IS_UPDATED;
    bool CRT_COMPUTED;
    bool ICRT_COMPUTED;
    bool TRANSF_COMPUTED;
    bool ITRANSF_COMPUTED;

    #ifdef CUFFTMUL_TRANSFORM
    Complex *d_polyTransf = 0x0;
    #endif

  protected:
    std::vector<ZZ> coefs;

    // Must be initialized on CRTSPACING definition and updated by crt(), if needed
    cuyasheint_t *d_polyCRT = 0x0;

    ZZ mod;
    Polynomial *phi;
    cudaStream_t stream;

};
#endif
