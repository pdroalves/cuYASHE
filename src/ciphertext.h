#ifndef CIPHERTEXT_H
#define CIPHERTEXT_H

#include "polynomial.h"

class Ciphertext: public Polynomial{
  public:
    Ciphertext operator+(Ciphertext &b);
    Ciphertext operator+(Polynomial &b);
    Ciphertext operator+=(Ciphertext &b);
    //   this->set_device_crt_residues( ((*this)+b).get_device_crt_residues());
    //   return *this;
    // }
    Ciphertext operator+=(Polynomial &b);
    Ciphertext operator*(Ciphertext &b);
    void convert();
    Ciphertext operator=(Polynomial p){
      level = 0;
      /////////////////////////////
      // Doesn't duplicate data //
      /////////////////////////////
      ///
      this->stream = p.get_stream();

      this->CRTSPACING = p.get_crt_spacing();
      this->d_bn_coefs = p.d_bn_coefs;
      this->h_bn_coefs = p.h_bn_coefs;
      this->d_polyCRT = p.get_device_crt_residues();

      this->set_crt_computed(p.get_crt_computed());
      this->set_icrt_computed(p.get_icrt_computed());
      this->set_host_updated(p.get_host_updated());

      if(p.get_host_updated())
        set_coeffs(p.get_coeffs());

      return *this;
    } 
    // Constructors
    Ciphertext(Polynomial *p){
        this->copy(*p);
    }
    Ciphertext(Polynomial p){
        this->copy(p);
    }
    Ciphertext(){
      this->set_stream();
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      if(Polynomial::global_mod > 0)
        // If a global mod is defined, use it
        this->mod = Polynomial::global_mod; // Doesn't copy. Uses the reference.
      
      if(Polynomial::global_phi){
        // If a global phi is defined, use it
        this->phi = Polynomial::global_phi; // Doesn't copy. Uses the reference.

        // If the irreductible polynomial have degree N, this polynomial's degree will be limited to N-1
        if(Polynomial::global_phi->deg() >= 0)
          this->update_crt_spacing(Polynomial::global_phi->deg()+1);
      }
      

      if(Polynomial::phi_set)
        this->coefs.resize(this->get_phi().deg()+1);
      

      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Ciphertext(ZZ p){
      this->set_stream();
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      this->mod = ZZ(p);// Copy

      if(Polynomial::global_phi){
      //   // If a global phi is defined, use it
        this->phi = Polynomial::global_phi; // Doesn't copy. Uses the reference.
        if(Polynomial::global_phi->deg() >= 0)
          this->update_crt_spacing(Polynomial::global_phi->deg()+1);
      }

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod << " but no phi."<< std::endl;
      #endif
    }
    Ciphertext(ZZ p,Polynomial P){
      this->set_stream();
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      this->mod = ZZ(p);// Copy
      *(this->phi) = Polynomial(P);// Copy

      // CRT Spacing should store the expected number of coefficients
      // If the irreductible polynomial have degree N, this polynomial's degree will be limited to N-1
      if(Polynomial::global_phi->deg() >= 0)
          this->update_crt_spacing(Polynomial::global_phi->deg()+1);
      
      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this-> phi << std::endl;
      #endif
    }
    Ciphertext(int spacing){
      this->set_stream();
      this->set_host_updated(true);
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
      this->update_crt_spacing(spacing);
      
      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }

      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Ciphertext(ZZ p,Polynomial P,int spacing){
      this->set_stream();
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      this->mod = ZZ(p);// Copy
      this->phi = &P;// Copy

      // std::cout << this->get_phi().to_string() << std::endl;

      // CRT Spacing set to spacing
      this->update_crt_spacing(spacing);

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this->phi << std::endl;
      #endif
    }
    Ciphertext(ZZ p,int spacing){
      this->set_stream();
      this->set_host_updated(true);
      this->set_crt_computed(false);
      this->set_icrt_computed(true);
      this->mod = ZZ(p);// Copy

      // CRT Spacing set to spacing
      this->update_crt_spacing(spacing);

      if(Polynomial::phi_set){
        this->coefs.resize(this->get_phi().deg()+1);
      }
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod << std::endl;
      #endif
    }

    void copy(Polynomial p){
        level = 0;
        Polynomial::copy(p);
    }

    int level = 0;
    bool aftermul = false;

  private:
    void keyswitch();
};

#endif
