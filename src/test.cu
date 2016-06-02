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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cuYASHE
#include <boost/test/unit_test.hpp>
#include <NTL/ZZ_pEX.h>

#include "polynomial.h"
#include "ciphertext.h"
#include "yashe.h"
#include "settings.h"
#include "cuda_functions.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#define NTESTS 10

struct CUDASuite
{
  // Test CUDA functions
    CUDASuite(){

    }

    ~CUDASuite()
    {
        BOOST_TEST_MESSAGE("teardown mass");
        cudaDeviceReset();
    }
};

struct PolySuite
{
  // Test polynomial functions
  ZZ q;
  int degree;
  Polynomial phi;
  ZZ_pX NTL_Phi;


    PolySuite(){
        BOOST_TEST_MESSAGE("setup PolySuite");
	      std::cout << "CRT Primes of " << CRTPRIMESIZE << " bits" << std::endl;

        degree = 16;

        q = conv<ZZ>("77287149995192912462927307869L");
        Polynomial::global_mod = q; // Defines default GF(q)
        Polynomial::BuildNthCyclotomic(&phi,degree);
        phi.set_mod(Polynomial::global_mod);
        Polynomial::global_phi = &phi;

        srand (36251);

        ZZ_p::init(Polynomial::global_mod);
        for(int i = 0; i <= phi.deg();i++){
          NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
        }
        ZZ_pE::init(NTL_Phi);

        Polynomial::gen_crt_primes(Polynomial::global_mod,degree);
        CUDAFunctions::init(degree);
        std::cout << "M = " << Polynomial::CRTProduct << std::endl;
        std::cout << std::endl;
    }

    ~PolySuite()
    {
        BOOST_TEST_MESSAGE("teardown mass");
        cudaDeviceReset();
    }
};


struct YasheSuite
{
    cuyasheint_t t;
    Yashe cipher;
    int degree;
    Polynomial phi;
    ZZ_pX NTL_Phi;


    YasheSuite()
    {
        BOOST_TEST_MESSAGE("setup YasheNonBinarySuite");
        srand (36251);

        // Params
        ZZ q;
        q = conv<ZZ>("77287149995192912462927307869L");
        // q = conv<ZZ>("655615111");
        Polynomial::global_mod = q;
        ZZ_p::init(q); // Defines GF(q)

        t = 35951;
        degree = 4;
        int w = 72;

        Polynomial::BuildNthCyclotomic(&phi, degree); // generate an cyclotomic polynomial
        #ifdef VERBOSE
        std::cout << "Cyclotomic polynomial: " << phi.to_string() << std::endl;
        #endif
        phi.set_mod(Polynomial::global_mod);
        Polynomial::global_phi = &phi;

        // Set params to NTL (just for comparison reasons)
        ZZ_p::init(Polynomial::global_mod);
        for(int i = 0; i <= phi.deg();i++){
          NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
        }
        ZZ_pE::init(NTL_Phi);


        Polynomial::gen_crt_primes(Polynomial::global_mod,degree);
        CUDAFunctions::init(degree);

        // Yashe
        cipher = Yashe();

        Yashe::d = degree;
        Yashe::phi = phi;
        Yashe::q = q;
        // std::cout << ZZ_p::modulus() << std::endl;
        // std::cout << q << std::endl;

        Yashe::t = t;
        Yashe::w = w;
        Yashe::lwq = floor(NTL::log(q)/NTL::log(to_ZZ(w)))+1;

        cipher.generate_keys();

        BOOST_TEST_MESSAGE("Set!");
    }

    ~YasheSuite()
    {
        BOOST_TEST_MESSAGE("teardown mass");
    }
};

BOOST_FIXTURE_TEST_SUITE(PolyFixture, PolySuite)

BOOST_AUTO_TEST_CASE(justCRT)
{
  Polynomial a;

  for(int count = 0; count < NTESTS;count ++){
    Polynomial::random(&a,degree-1);

    Polynomial b;
    b.copy(a);//Copy
    b.update_device_data();
    b.set_host_updated(false);

    #ifdef VERBOSE
    std::cout << std::endl << a.to_string() << " == " << std::endl<< b.to_string() << std::endl;
    #endif
    BOOST_REQUIRE(a == b);
  }
}


BOOST_AUTO_TEST_CASE(auxLessThan)
{
  // 32 bits
  for(int i = 0; i < NTESTS; i++){
    uint64_t a = rand();
    uint64_t b = rand();
    BOOST_REQUIRE((a < b) == lessThan(a,b));
  }
  // 64 bits
  for(int i = 0; i < NTESTS; i++){
    uint64_t a = ((long long)rand() << 32) | rand();
    uint64_t b = ((long long)rand() << 32) | rand();
    BOOST_REQUIRE((a < b) == lessThan(a,b));
  }
}

BOOST_AUTO_TEST_CASE(simpleAdd)
{

  Polynomial a;
  Polynomial::random(&a,degree-1);

  Polynomial b(a);//Copy
  Polynomial c = a+b;
  c.update_host_data();//
  c.normalize();

  // #ifdef VERBOSE
  std::cout << "a: " << a.to_string() << std::endl;
  std::cout << "(2*a): " << (a*2).to_string() << std::endl;
  std::cout << "b: " << b.to_string() << std::endl;
  std::cout << "c: " << c.to_string() << std::endl;
  // #endif
  BOOST_REQUIRE(c == (a*2));
}

BOOST_AUTO_TEST_CASE(multipleAdds)
{

  Polynomial a;
  Polynomial::random(&a,degree-1);

  Polynomial b;
  for(int count = 0; count < NTESTS;count++){
    b += a;
    // std::cout << "b: " << b.to_string() << std::endl;
  }

  #ifdef VERBOSE
  std::cout << "a: " << a.to_string() << std::endl;
  std::cout << "b: " << b.to_string() << std::endl;
  #endif
  BOOST_REQUIRE(b == (a*NTESTS)%(b.get_mod()));
}

BOOST_AUTO_TEST_CASE(zeroAdd)
{

  Polynomial a;
  Polynomial b;
  Polynomial::random(&a,degree-1);


  Polynomial r = a + b;
  #ifdef VERBOSE
  std::cout << "M: "<<Polynomial::CRTProduct << std::endl;
  std::cout << "a: " << a.to_string() <<std::endl;
  std::cout << "b: " << b.to_string() <<std::endl;
  std::cout << "r: " << r.to_string() << std::endl;
  #endif
  
  BOOST_REQUIRE(r == a);

}
BOOST_AUTO_TEST_CASE(getDegreeDevice)
{ 

  ///////////////
  // On device //
  ///////////////
  for(int i = 0; i < NTESTS; i++){
    int degree = rand()%NTESTS;
    Polynomial a;
    Polynomial::random(&a,degree);
    
    a.update_device_data();
    a.set_host_updated(false);
    BOOST_REQUIRE(a.deg() == degree);
  }
}

BOOST_AUTO_TEST_CASE(getDegreeHost)
{
  /////////////
  // On host //
  /////////////
  for(int i = 0; i < NTESTS; i++){
    int degree = rand()%NTESTS;
    Polynomial a;
    Polynomial::random(&a,degree);
    BOOST_REQUIRE( a.deg() == degree);
  }
}

#ifdef NTTMUL
BOOST_AUTO_TEST_CASE(wNTest)
{
  cuyasheint_t *W;
  cuyasheint_t *WInv;
  int N = degree;

  W = (cuyasheint_t*)malloc(N*sizeof(cuyasheint_t));
  WInv = (cuyasheint_t*)malloc(N*sizeof(cuyasheint_t));

  cudaError_t result = cudaMemcpy(W,CUDAFunctions::d_W , N*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
  BOOST_REQUIRE(result == cudaSuccess);
  result = cudaMemcpy(WInv,CUDAFunctions::d_WInv , N*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
  BOOST_REQUIRE(result == cudaSuccess);

  cuyasheint_t wN = CUDAFunctions::wN;
  ZZ PZZ = conv<ZZ>("18446744069414584321L");
  cuyasheint_t k = conv<cuyasheint_t>(PZZ-1)/N;
  ZZ wNZZ = NTL::PowerMod(ZZ(7),k,PZZ);
  std::cout << wNZZ << std::endl;

  BOOST_REQUIRE(wNZZ == NTL::to_ZZ(wN));
  for(int i = 0; i < N;i++)
    BOOST_REQUIRE(NTL::MulMod(NTL::to_ZZ(W[i]),NTL::to_ZZ(WInv[i]),PZZ) == 1);

}
BOOST_AUTO_TEST_CASE(smulTest)
{
  const uint64_t P = 18446744069414584321L;
  for(int i = 0; i < 100*NTESTS;i++){
    uint64_t a = (((long long)rand() << 32) | rand());
    uint64_t b = (((long long)rand() << 32) | rand());
    uint64_t result = s_mul(a,b);
    BOOST_REQUIRE(result == ((__uint128_t)(a)*(__uint128_t)(b) % P));
  }
  for(int i = 0; i < 100*NTESTS;i++){
    uint64_t a = rand();
    uint64_t b = rand();
    uint64_t result = s_mul(a,b);
    BOOST_REQUIRE(result == ((__uint128_t)(a)*(__uint128_t)(b) % P));
  }

}
#endif


BOOST_AUTO_TEST_CASE(simpleMultiplyByPolynomial)
{
  Polynomial a;
  Polynomial b;

  a.set_coeffs(degree);
  b.set_coeffs(degree);
  for(int i = 0; i < degree;i++){
    a.set_coeff(i,i);
    b.set_coeff(i,1);
  }

  ZZ_pEX b_ntl;
  ZZ_pEX a_ntl;
  for(int i = 0;i <= a.deg();i++)
    NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
  for(int i = 0;i <= b.deg();i++)
    NTL::SetCoeff(b_ntl,i,conv<ZZ_p>(b.get_coeff(i)));

  Polynomial c = a*b;
  c.reduce();
  c.normalize();

  ZZ_pEX c_ntl = a_ntl*b_ntl;
  c_ntl %= conv<ZZ_pEX>(NTL_Phi);

  // #ifdef DEBUG
  std::cout << "degree = " << degree << std::endl;
  std::cout << "CUDAFunctions::N = " << CUDAFunctions::N << std::endl;
    std::cout << "a: " << a.to_string() << " degree: " << a.deg() <<std::endl;
    std::cout << "b: " << b.to_string() << " degree: " << b.deg() <<std::endl;
    std::cout << "c: " << c.to_string() << " degree: " << c.deg() <<std::endl;
    std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
  // #endif

  BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
  for(int i = 0;i <= c.deg();i++){

    ZZ ntl_value;
    if( NTL::IsZero(NTL::coeff(c_ntl,i)) )
    // Without this, NTL raises an exception when we call rep()
      ntl_value = 0L;
    else
      ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(c_ntl,i))[0]);

    BOOST_REQUIRE(c.get_coeff(i) == ntl_value);
  }

}

BOOST_AUTO_TEST_CASE(multiplyByPolynomial)
{

  CUDAFunctions::init(degree);

  for(int count = 0; count < NTESTS; count++){

    Polynomial a;
    Polynomial b;

    Polynomial::random(&a,degree-1);
    Polynomial::random(&b,degree-1);

    ZZ_pEX b_ntl;
    ZZ_pEX a_ntl;
    for(int i = 0;i <= a.deg();i++)
      NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
    for(int i = 0;i <= b.deg();i++)
      NTL::SetCoeff(b_ntl,i,conv<ZZ_p>(b.get_coeff(i)));

    Polynomial c = a*b;
      // std::cout << "c: " << (c.to_string()) <<std::endl;
    c.reduce();

    ZZ_pEX c_ntl = a_ntl*b_ntl;
    c_ntl %= conv<ZZ_pEX>(NTL_Phi);

      // std::cout << "a: " << (a.to_string()) <<std::endl;
      // std::cout << "b: " << (b.to_string()) <<std::endl;
      // std::cout << "c: " << (c.to_string()) <<std::endl;
      // std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
    #ifdef DEBUG
    if(c_ntl != c){
    }
    std::cout << "count: " << count << std::endl;
    #endif
      // std::cout << "c: " << c.to_string() << " degree: " << c.deg() <<std::endl;
      // std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;

    BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
    for(int i = 0;i <= c.deg();i++){
      ZZ ntl_value;
      if( NTL::IsZero(NTL::coeff(c_ntl,i)) )
      // Without this, NTL raises an exception when we call rep()
        ntl_value = 0L;
      else
        ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(c_ntl,i))[0]);

      BOOST_REQUIRE(c.get_coeff(i) == ntl_value);
    }

  }
}


BOOST_AUTO_TEST_CASE(multiplyAndAddByPolynomial)
{

  CUDAFunctions::init(degree);

  for(int count = 0; count < NTESTS; count++){

    Polynomial a;
    Polynomial b;

    Polynomial::random(&a,degree-1);
    Polynomial::random(&b,degree-1);

    ZZ_pEX b_ntl;
    ZZ_pEX a_ntl;
    for(int i = 0;i <= a.deg();i++)
      NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
    for(int i = 0;i <= b.deg();i++)
      NTL::SetCoeff(b_ntl,i,conv<ZZ_p>(b.get_coeff(i)));

    Polynomial c = a*b+a;
    c.reduce();

    ZZ_pEX c_ntl = a_ntl*b_ntl+a_ntl;
    c_ntl %= conv<ZZ_pEX>(NTL_Phi);

    #ifdef DEBUG
    if(c_ntl != c){
      std::cout << "a: " << a << " degree: " << NTL::deg(a) <<std::endl;
      std::cout << "b: " << b << " degree: " << NTL::deg(b) <<std::endl;
      std::cout << "c: " << c << " degree: " << NTL::deg(c) <<std::endl;
      std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
    }
    std::cout << "count: " << count << std::endl;
    #endif
      // std::cout << "c: " << c.to_string() << " degree: " << c.deg() <<std::endl;
      // std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;

    BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
    for(int i = 0;i <= c.deg();i++){
      ZZ ntl_value;
      if( NTL::IsZero(NTL::coeff(c_ntl,i)) )
      // Without this, NTL raises an exception when we call rep()
        ntl_value = 0L;
      else
        ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(c_ntl,i))[0]);

      BOOST_REQUIRE(c.get_coeff(i) == ntl_value);
    }

  }
}

BOOST_AUTO_TEST_CASE(addAndMultiplyByPolynomial)
{

  CUDAFunctions::init(degree);

  for(int count = 0; count < NTESTS; count++){

    Polynomial a;
    Polynomial b;

    Polynomial::random(&a,degree-1);
    Polynomial::random(&b,degree-1);

    ZZ_pEX b_ntl;
    ZZ_pEX a_ntl;
    for(int i = 0;i <= a.deg();i++)
      NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
    for(int i = 0;i <= b.deg();i++)
      NTL::SetCoeff(b_ntl,i,conv<ZZ_p>(b.get_coeff(i)));

    Polynomial c = a*b+a;
    c.reduce();

    ZZ_pEX c_ntl = a_ntl*b_ntl+a_ntl;
    c_ntl %= conv<ZZ_pEX>(NTL_Phi);

    #ifdef DEBUG 
     std::cout << "a: " << a.to_string() <<std::endl;
      std::cout << "b: " << b.to_string() <<std::endl;
      std::cout << "c: " << c.to_string() <<std::endl;
      std::cout << "c_ntl: " << c_ntl <<std::endl<<std::endl;
    #endif

    // std::cout << NTL::deg(c_ntl) << " == " << c.deg() << std::endl;;
    BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
    for(int i = 0;i <= c.deg();i++){
      ZZ ntl_value;
      if( NTL::IsZero(NTL::coeff(c_ntl,i)) )
      // Without this, NTL raises an exception when we call rep()
        ntl_value = 0L;
      else
        ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(c_ntl,i))[0]);

      BOOST_REQUIRE(c.get_coeff(i) == ntl_value);
    }

  }
}

BOOST_AUTO_TEST_CASE(modularInversion)
{

  Polynomial a;
  Polynomial::random(&a,Polynomial::global_phi->deg()-1);

  Polynomial aInv = Polynomial::InvMod(a,Polynomial::global_phi);
  
  Polynomial result = a*aInv;
  
  // std::cout << "a: " << a.to_string() << std::endl;
  // std::cout << "aInv: " << aInv.to_string() << std::endl;
  // std::cout << "result before reduce: " << result.to_string() << std::endl;
  result.reduce();
  // std::cout << "result after reduce: " << result.to_string() << std::endl;

  Polynomial one = Polynomial();
  one.set_coeff(0,1);

  result.normalize();

  BOOST_REQUIRE(result == one);

}

BOOST_AUTO_TEST_CASE(severalMultiplications)
{
  Polynomial a;
  Polynomial b;

  CUDAFunctions::init(degree);

  Polynomial::random(&a,degree-1);
  Polynomial::random(&b,degree-1);

  ZZ_pEX a_ntl;
  ZZ_pEX b_ntl;
  
  for(int i = 0;i <= a.deg();i++)
    NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
  for(int i = 0;i <= b.deg();i++)
    NTL::SetCoeff(b_ntl,i,conv<ZZ_p>(b.get_coeff(i)));

  // std::cout << "Iteration "<< 0 << std::endl;

  Polynomial c = a*b;
  c.reduce(); // %phi %q
    
  ZZ_pEX c_ntl = a_ntl*b_ntl;
  c_ntl %= conv<ZZ_pEX>(NTL_Phi);
  
  #ifdef DEBUG
  std::cout << "Iteration 0" << std::endl;
  std::cout << "a: " << a.to_string() << " degree: " << a.deg() <<std::endl;
  std::cout << "b: " << b.to_string() << " degree: " << b.deg() <<std::endl;
  std::cout << "c: " << c.to_string() << " degree: " << c.deg() <<std::endl;
  std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
  #endif
   
  BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
  for(int i = 0;i <= c.deg();i++){

    ZZ ntl_value;
    if( NTL::IsZero(NTL::coeff(c_ntl,i)) )
    // Without this, NTL raises an exception when we call rep()
    ntl_value = 0L;
    else
    ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(c_ntl,i))[0]);

    BOOST_REQUIRE(c.get_coeff(i) == ntl_value);
  }
  c.update_host_data();
  c.set_icrt_computed(false);
  for(unsigned int i = 1; i < NTESTS; i++){
    c = c*a;
    c.reduce();

    c_ntl = c_ntl*a_ntl;
    c_ntl %= conv<ZZ_pEX>(NTL_Phi);

    #ifdef DEBUG
    std::cout << "Iteration "<< i << std::endl;
    std::cout << "a: " << a.to_string() << " degree: " << a.deg() <<std::endl;
    std::cout << "b: " << b.to_string() << " degree: " << b.deg() <<std::endl;
    std::cout << "c: " << c.to_string() << " degree: " << c.deg() <<std::endl;
    std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
    #endif

    BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
    for(int i = 0;i <= c.deg();i++){

      ZZ ntl_value;
      if( NTL::IsZero(NTL::coeff(c_ntl,i)) )
      // Without this, NTL raises an exception when we call rep()
      ntl_value = 0L;
      else
      ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(c_ntl,i))[0]);

      BOOST_REQUIRE(c.get_coeff(i) == ntl_value);
    }
  }

  a.release();
  b.release();
  c.release();
}
BOOST_AUTO_TEST_CASE(phiReduceCPU)
{
  //CPU
  for(int count = 0; count < NTESTS; count++){

    Polynomial a;
    Polynomial::random(&a,2*degree-2);

    ZZ_pEX a_ntl;
    for(int i = 0;i <= a.deg();i++)
      NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));

    a.reduce();
    a %= Polynomial::global_mod;
    a_ntl %= conv<ZZ_pEX>(ZZ_pE::modulus());

    // std::cout << a.to_string() << std::endl;
    // std::cout << a_ntl << std::endl;
    BOOST_REQUIRE(NTL::deg(a_ntl) == a.deg());
    for(int i = 0;i <= a.deg();i++){

      ZZ ntl_value;
      if( NTL::IsZero(NTL::coeff(a_ntl,i)) )
      // Without this, NTL raises an exception when we call rep()
        ntl_value = 0L;
      else
        ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(a_ntl,i))[0]);

      BOOST_REQUIRE(a.get_coeff(i) == ntl_value);
    }
  }
}

BOOST_AUTO_TEST_CASE(phiReduceGPU)
{
  //GPU
  for(int count = 0; count < NTESTS; count++){

    Polynomial a;
    Polynomial::random(&a,2*degree-2);

    ZZ_pEX a_ntl;
    for(int i = 0;i <= a.deg();i++)
      NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
    // std::cout << "a_ntl = " << a_ntl << std::endl;;
    // std::cout << "a = " << a.to_string() << std::endl;;

    a.update_device_data();
    a.set_host_updated(false);

    a.reduce();
    a_ntl %= conv<ZZ_pEX>(ZZ_pE::modulus());

    // std::cout << "a_ntl = " << a_ntl << std::endl;;
    // std::cout << "a = " << a.to_string() << std::endl;;

    BOOST_REQUIRE(NTL::deg(a_ntl) == a.deg());
    for(int i = 0;i <= a.deg();i++){

      ZZ ntl_value;
      if( NTL::IsZero(NTL::coeff(a_ntl,i)) )
      // Without this, NTL raises an exception when we call rep()
        ntl_value = 0L;
      else
        ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(a_ntl,i))[0]);
      BOOST_REQUIRE(a.get_coeff(i) == ntl_value);
    }
  }
}


BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(YasheFixture, YasheSuite)

BOOST_AUTO_TEST_CASE(encryptDecrypt)
{
  Polynomial a;

  for(int i = 0; i < NTESTS;i++){
    a.set_coeff(0,conv<ZZ>(rand())%t);

    Ciphertext c = cipher.encrypt(a);
    Polynomial a_decrypted = cipher.decrypt(c);

    // #ifdef VERBOSE
    std::cout << "a: " << a.to_string() << std::endl;
    std::cout << "a_decrypted: " << a_decrypted.to_string() << std::endl;
    // #endif
    BOOST_REQUIRE( a_decrypted == a);
  }
}


BOOST_AUTO_TEST_CASE(encryptandAdd)
{
  Polynomial a;

  for(int i = 0; i < NTESTS;i++){
    a.set_coeff(0,conv<ZZ>(rand())%t);

    Ciphertext c = cipher.encrypt(a);

    c = c + c;

    Polynomial a_decrypted = cipher.decrypt(c);

    Polynomial value = (a+a);
    value.update_host_data();
    Polynomial value_reduced = value % t;

    #ifdef VERBOSE
    std::cout << "Original: " << a.to_string() << std::endl;
    std::cout << "Original *2: " << ((a+a)%t).to_string() << std::endl;
    std::cout << "Decrypted: " << a_decrypted.to_string() << std::endl;
    #endif

    BOOST_REQUIRE( a_decrypted == value_reduced);
  }
}

BOOST_AUTO_TEST_CASE(encryptandMul)
{
  Polynomial a;

  for(int i = 0; i < NTESTS;i++){
    a.set_coeff(0,conv<ZZ>(rand())%t);
    // std::cout << "Value: " << a.to_string() << std::endl;

    Ciphertext c1 = cipher.encrypt(a);
    Ciphertext c2 = cipher.encrypt(a);
    Ciphertext c = c1*c2;

    Polynomial a_decrypted = cipher.decrypt(c);

    Polynomial value = (a*a);
    Polynomial value_reduced = value % t;

    #ifdef VERBOSE
    std::cout << "Original: " << a.to_string() << std::endl;
    std::cout << "Original *2: " << ((a*a)%t).to_string() << std::endl;
    std::cout << "value: " << value.to_string() << std::endl;
    std::cout << "value_reduced: " << value_reduced.to_string() << std::endl;
    std::cout << "Decrypted: " << a_decrypted.to_string() << std::endl;
    #endif

    BOOST_REQUIRE( a_decrypted == value_reduced);
  }
}

BOOST_AUTO_TEST_SUITE_END()
