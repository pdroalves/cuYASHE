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
#include <NTL/ZZ_pEX.h>

#include "polynomial.h"
#include "settings.h"
#include "cuda_functions.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

int main(){
  int degree;
  Polynomial phi;

  degree = 8;

  Polynomial::global_mod = conv<ZZ>("655615111"); // Defines default GF(q)
  Polynomial::BuildNthCyclotomic(&phi,degree);
  phi.set_mod(Polynomial::global_mod);
  Polynomial::global_phi = &phi;

  srand (36251);

  // Polynomial::gen_crt_primes(Polynomial::global_mod,degree);

  Polynomial::CRTPrimes.clear();
  Polynomial::CRTMpi.clear();  
  Polynomial::CRTInvMpi.clear();

  Polynomial::CRTPrimes.push_back(751);
  Polynomial::CRTPrimes.push_back(839);
  Polynomial::CRTPrimes.push_back(829);
  Polynomial::CRTPrimes.push_back(661);
  Polynomial::CRTPrimes.push_back(797);
  Polynomial::CRTPrimes.push_back(1019);
  Polynomial::CRTPrimes.push_back(857);
  Polynomial::CRTProduct = to_ZZ("240309652370522267791");

  // Compute M/pi and it's inverse
  for(unsigned int i = 0; i < Polynomial::CRTPrimes.size();i++){
    ZZ pi = to_ZZ(Polynomial::CRTPrimes[i]);
    Polynomial::CRTMpi.push_back(Polynomial::CRTProduct/pi);
    Polynomial::CRTInvMpi.push_back(conv<cuyasheint_t>(NTL::InvMod(Polynomial::CRTMpi[i]%pi,pi)));
  }
  
  CUDAFunctions::write_crt_primes();
  CUDAFunctions::init(16);

  ZZ_p::init(Polynomial::global_mod);

  std::cout << "Phi: " << phi.to_string() << std::endl;
  std::cout << "M: " << Polynomial::CRTProduct << std::endl;

  // GPU
  Polynomial a;
  a.set_coeff(0,1);
  a.set_coeff(1,1);
  a.set_coeff(2,1);
  a.set_coeff(3,1);
  a.set_coeff(4,1);
  a.set_coeff(5,1);
  a.set_coeff(6,1);
  a.set_coeff(7,1);
  a.set_coeff(8,1);

  a.update_device_data();
  a.set_host_updated(false);
  a.reduce();

  std::cout << "GPU: " <<a.to_string() << std::endl;

  // CPU
  a.set_coeff(0,1);
  a.set_coeff(1,1);
  a.set_coeff(2,1);
  a.set_coeff(3,1);
  a.set_coeff(4,1);
  a.set_coeff(5,1);
  a.set_coeff(6,1);
  a.set_coeff(7,1);
  a.set_coeff(8,1);

  a.set_crt_computed(false);
  a.set_icrt_computed(false);
  a.set_host_updated(true);
  a.reduce();

  std::cout << "CPU: " <<a.to_string() << std::endl;

 // GPU
  a.set_coeff(0,722); 
  a.set_coeff(1,173); 
  a.set_coeff(2,735); 
  a.set_coeff(3,651); 
  a.set_coeff(4,460); 
  a.set_coeff(5,161); 
  a.set_coeff(6, 56); 
  a.set_coeff(7, 80); 
  a.set_coeff(8,762); 
  a.set_coeff(9,  9); 
  a.set_coeff(10,220); 
  a.set_coeff(11,281);  
  a.set_coeff(12, 62);  
  a.set_coeff(13, 96);  
  a.set_coeff(14,239);  

  a.update_device_data();
  a.set_host_updated(false);
  a.reduce();
  // a %= Polynomial::global_mod;

  std::cout << "GPU: " <<a.to_string() << std::endl;

  // CPU
  a.set_coeff(0,722); 
  a.set_coeff(1,173); 
  a.set_coeff(2,735); 
  a.set_coeff(3,651); 
  a.set_coeff(4,460); 
  a.set_coeff(5,161); 
  a.set_coeff(6, 56); 
  a.set_coeff(7, 80); 
  a.set_coeff(8,762); 
  a.set_coeff(9,  9); 
  a.set_coeff(10,220); 
  a.set_coeff(11,281);  
  a.set_coeff(12, 62);  
  a.set_coeff(13, 96);  
  a.set_coeff(14,239);  

  a.set_icrt_computed(false);
  a.set_crt_computed(false);
  a.set_host_updated(true);
  a.reduce();
  // a %= Polynomial::global_mod;

  std::cout << "CPU: " <<a.to_string() << std::endl;

   // GPU
  a.set_coeffs();
  a.set_coeff(0,to_ZZ("100841831415836174481434746063997936280"));
  ZZ m = to_ZZ("77287149995192912462927307869L");

  a.set_icrt_computed(false);
  a.set_crt_computed(false);
  a.set_host_updated(true);

  a.update_device_data();
  a.set_host_updated(false);
  a.modn(m);

  std::cout << "GPU: " <<a.to_string() << std::endl;

  // CPU
  a.set_coeffs(1);
  a.set_coeff(0,to_ZZ("100841831415836174481434746063997936280"));

  a.set_icrt_computed(false);
  a.set_crt_computed(false);
  a.set_host_updated(true);

  a %= m;

  std::cout << "CPU: " <<a.to_string() << std::endl;


  // GPU
  a.set_coeffs();
  a.set_coeff(0,to_ZZ("61841506593550295")), 
  a.set_coeff(1,to_ZZ("106916833401539892")), 
  a.set_coeff(2,to_ZZ("757073061970113119")), 
  a.set_coeff(3,to_ZZ("3338412596642650852")), 
  a.set_coeff(4,to_ZZ("5470565564447407791")), 
  a.set_coeff(5,to_ZZ("5548103536799580232")), 
  a.set_coeff(6,to_ZZ("6089003692894423880")),
  a.set_coeff(7,to_ZZ("7733984529407217518")), 
  a.set_coeff(8,to_ZZ("8022978165266350588")), 
  a.set_coeff(9,to_ZZ("9170489421306427724")), 
  a.set_coeff(10,to_ZZ("8993977458886462143")), 
  a.set_coeff(11,to_ZZ("6882188452282780506")), 
  a.set_coeff(12,to_ZZ("4376377560524434132")), 
  a.set_coeff(13,to_ZZ("2285160538342347456")), 
  a.set_coeff(14,to_ZZ("1122693488336169022"));

  a.set_icrt_computed(false);
  a.set_crt_computed(false);
  a.set_host_updated(true);

  a.update_device_data();
  a.set_host_updated(false);
  a.modn(m);


}
