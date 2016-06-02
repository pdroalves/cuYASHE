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
      #ifdef CUFFTMUL_TRANSFORM
      this->set_device_transf_residues( p.get_device_transf_residues() );
      #endif

      this->set_crt_computed(p.get_crt_computed());
      this->set_icrt_computed(p.get_icrt_computed());
      this->set_host_updated(p.get_host_updated());
      this->set_transf_computed(p.get_transf_computed());
      this->set_itransf_computed(p.get_itransf_computed());

      if(p.get_host_updated())
        set_coeffs(p.get_coeffs());

      return *this;
    } 
    // Constructors
    Ciphertext(Polynomial *p){
      level = 0;
      /////////////////////////////
      // Doesn't duplicate data //
      /////////////////////////////
      ///
      this->stream = p->get_stream();

      this->CRTSPACING = p->get_crt_spacing();
      this->d_bn_coefs = p->d_bn_coefs;
      this->h_bn_coefs = p->h_bn_coefs;
      this->d_polyCRT = p->get_device_crt_residues();
      #ifdef CUFFTMUL_TRANSFORM
      this->set_device_transf_residues( p->get_device_transf_residues() );
      #endif

      this->set_crt_computed(p->get_crt_computed());
      this->set_icrt_computed(p->get_icrt_computed());
      this->set_host_updated(p->get_host_updated());
      this->set_transf_computed(p->get_transf_computed());
      this->set_itransf_computed(p->get_itransf_computed());

      if(p->get_host_updated())
        set_coeffs(p->get_coeffs());

    }
    Ciphertext(Polynomial p){
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
      #ifdef CUFFTMUL_TRANSFORM
      this->set_device_transf_residues( p.get_device_transf_residues() );
      #endif

      this->set_crt_computed(p.get_crt_computed());
      this->set_icrt_computed(p.get_icrt_computed());
      this->set_host_updated(p.get_host_updated());
      this->set_transf_computed(p.get_transf_computed());
      this->set_itransf_computed(p.get_itransf_computed());

      if(p.get_host_updated())
        set_coeffs(p.get_coeffs());

    }
    Ciphertext(){
      this->set_stream();
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
      this->set_icrt_computed(false);
      this->set_transf_computed(false);
      this->set_itransf_computed(false);
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
      this->set_icrt_computed(false);
      this->set_transf_computed(false);
      this->set_itransf_computed(false);
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

      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Ciphertext(ZZ p,Polynomial P,int spacing){
      this->set_stream();
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
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this->phi << std::endl;
      #endif
    }
    Ciphertext(ZZ p,int spacing){
      this->set_stream();
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
    void keyswitch_mul(std::vector<Polynomial> *P);
};

#endif
