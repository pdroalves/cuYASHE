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
  ZZ_pX NTL_Phi;

  ////////////////
  // First test //
  ////////////////
  std::cout << "Prime size: " << CRTPRIMESIZE << std::endl;
  std::cout << "First test: " << std::endl;
  degree = 8;

  Polynomial::global_mod = conv<ZZ>("1171313591017775093490277364417"); // Defines default GF(q)
  Polynomial::BuildNthCyclotomic(&phi,degree);
  phi.set_mod(Polynomial::global_mod);
  Polynomial::global_phi = &phi;
// 
  srand (36251);

  Polynomial::gen_crt_primes(Polynomial::global_mod,degree);
  std::cout << "M = " << Polynomial::CRTProduct << std::endl;
  ZZ_p::init(Polynomial::global_mod);
  for(int i = 0; i <= phi.deg();i++){
    NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
  }
  ZZ_pE::init(NTL_Phi);

  Polynomial a,b,c;
  a.set_coeffs(8);
  a.set_coeff(0,1304768405);
  a.set_coeff(1,1912295543);
  a.set_coeff(2,584306160);
  a.set_coeff(3,476779113);
  a.set_coeff(4,2057832156);
  a.set_coeff(5,1274012458);
  a.set_coeff(6,1818971124);
  a.set_coeff(7,1943566066);

  b.set_coeffs(8);
  b.set_coeff(0,47396539);
  b.set_coeff(1,12477803);
  b.set_coeff(2,540722570);
  b.set_coeff(3,1743223311);
  b.set_coeff(4,1316384259);
  b.set_coeff(5,1278652539);
  b.set_coeff(6,635141178);
  b.set_coeff(7,577646167);

  c = a*b;
  
  std::cout << "a: " << a.to_string() << std::endl;
  std::cout << "b: " << b.to_string() << std::endl;
  std::cout << "a*b: " <<c.to_string() << std::endl;

  ZZ expected_result1[] = {to_ZZ("61841506593550295L"),
                             to_ZZ("106916833401539892L"),
                             to_ZZ("757073061970113119L"),
                             to_ZZ("3338412596642650852L"),
                             to_ZZ("5470565564447407791L"),
                             to_ZZ("5548103536799580232L"),
                             to_ZZ("6089003692894423880L"),
                             to_ZZ("7733984529407217518L"),
                             to_ZZ("8022978165266350588L"),
                             to_ZZ("9170489421306427724L"),
                             to_ZZ("8993977458886462143L"),
                             to_ZZ("6882188452282780506L"),
                             to_ZZ("4376377560524434132L"),
                             to_ZZ("2285160538342347456L"),
                             to_ZZ("1122693488336169022L")};
  c.normalize();
  assert(c.get_coeffs().size() == 15);

  for(int i = 0; i < c.get_coeffs().size();i++)
      assert(c.get_coeff(i) == expected_result1[i]);
  std::cout << "It works!\n" << std::endl;;

  /////////////////
  // Second test //
  /////////////////
  std::cout << std::endl << "Second test: " << std::endl;
  a.set_coeffs(8);
  a.set_coeff(0,1304768405);
  a.set_coeff(1,1912295543);
  a.set_coeff(2,584306160);
  a.set_coeff(3,476779113);
  a.set_coeff(4,2057832156);
  a.set_coeff(5,1274012458);
  a.set_coeff(6,1818971124);
  a.set_coeff(7,1943566066);
  
  b.set_coeffs(8);
  b.set_coeff(0,to_ZZ("13860721405712617825882419437"));
  b.set_coeff(1,to_ZZ("911979968073012785910226533"));
  b.set_coeff(2,to_ZZ("60134818568832314735148266017"));
  b.set_coeff(3,to_ZZ("30228208549061091796489949751"));
  b.set_coeff(4,to_ZZ("42496336825615196107672318132"));
  b.set_coeff(5,to_ZZ("2538357487121433195951809389"));
  b.set_coeff(6,to_ZZ("19401757472799351763805908821"));
  b.set_coeff(7,to_ZZ("68415359680988770693366249506"));

  c.set_coeffs();
  c = a*b;
  
  std::cout << "a: " << a.to_string() << std::endl;
  std::cout << "b: " << b.to_string() << std::endl;
  std::cout << "a*b: " <<c.to_string() << std::endl;

  ZZ expected_result2[] = { to_ZZ("18085031360681010249051172126355487985"),
                            to_ZZ("27695718415243509623513993478082759156"),
                            to_ZZ("88304891436672768443904444999435867224"),
                            to_ZZ("161577734953354918933760345835324042047"),
                            to_ZZ("177348142234080397165460901588496065374"),
                            to_ZZ("150446511525560453344956466169095450596"),
                            to_ZZ("219533428887126844535620298106253625669"),
                            to_ZZ("315527818352347626764175594181751346765"),
                            to_ZZ("380494624177053395698844189209855453911"),
                            to_ZZ("280450476250800780083801312965924383119"),
                            to_ZZ("211828603909665970332138017649830954350"),
                            to_ZZ("252717044998320508082871060193586798902"),
                            to_ZZ("127386722625350085506673181613116824226"),
                            to_ZZ("162154161142687163774421077602378932930"),
                            to_ZZ("132969771469154360046681933849550863396")};
  c.normalize();
  assert(c.get_coeffs().size() == 15);
  
  for(int i = 0; i < c.get_coeffs().size();i++)
      assert(c.get_coeff(i) == expected_result2[i]);
  std::cout << "It works!\n" << std::endl;;
  // a.release();
  // b.release();
  // c.release();

  cudaDeviceReset();
}
