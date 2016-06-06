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
#include "ciphertext.h"
#include "yashe.h"
#include "settings.h"
#include "cuda_functions.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

NTL_CLIENT


int main(void){
    uint64_t t;
    Yashe cipher;
    int degree;
    Polynomial phi;
    ZZ_pX NTL_Phi;

    srand (36251);

    //////////////
    // Settings //
    //////////////
    // Params
    ZZ q;
    q = conv<ZZ>("8191");
    // q = conv<ZZ>("655615111");
    Polynomial::global_mod = q;
    ZZ_p::init(q); // Defines GF(q)

    t = 17;
    degree = 4;
    int w = 72;

    Polynomial::BuildNthCyclotomic(&phi, degree); // generate an cyclotomic polynomial
    phi.set_mod(Polynomial::global_mod);
    Polynomial::gen_crt_primes(Polynomial::global_mod,degree);
    Polynomial::global_phi = &phi;
    ZZ_p::init(Polynomial::global_mod);
    for(int i = 0; i <= phi.deg();i++){
      NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
    }
    ZZ_pE::init(NTL_Phi);

    CUDAFunctions::init(degree);

    //////////////

    // Yashe
    cipher = Yashe();

    Yashe::d = degree;
    Yashe::phi = phi;
    Yashe::q = q;

    Yashe::t = t;
    Yashe::w = w;
    Yashe::lwq = floor(NTL::log(q)/(log(2)*w)+1);

    cipher.generate_keys();

    //////////
    Polynomial pA;
    pA.set_coeff(0,13);
    std::cout << "Plaintext: " << pA.to_string() << std::endl;

    Ciphertext cA = cipher.encrypt(pA);
    std::cout << "Ciphertext: " << cA.to_string() << std::endl;
    
    Polynomial result;
    result = cipher.decrypt(cA);
    std::cout << "result: " << result.to_string() << std::endl;

    // Ciphertext cB;
    // cB.set_coeff(0,7083);
    // cB.set_coeff(1,2131);
    // cB.set_coeff(2,3253);
    // cB.set_coeff(3,4611);

    // result = cipher.decrypt(cB);
    // std::cout << "result 2: " << result.to_string() << std::endl;

}
