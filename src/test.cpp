#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cuYASHE
#include <boost/test/unit_test.hpp>
#include <NTL/ZZ_pEX.h>

#include "polynomial.h"
#include "common.h"

#define NTESTS 100

struct PolySuite
{
    int degree;
    Polynomial phi;

    PolySuite(){
        BOOST_TEST_MESSAGE("setup PolySuite");

        degree = 128;

        Polynomial::global_mod = conv<ZZ>("1171313591017775093490277364417"); // Defines default GF(q)
        Polynomial::BuildNthCyclotomic(&phi,degree);
        // std::cout << phi.to_string() << std::endl;
        phi.set_mod(Polynomial::global_mod);
        Polynomial::global_phi = &phi;

        srand (36251);

        ZZ_p::init(Polynomial::global_mod);

        Polynomial::gen_crt_primes(Polynomial::global_mod,degree);
    }

    ~PolySuite()
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

    Polynomial b(a);//Copy
    a.crt();
    a.update_device_data();
    a.set_host_updated(false);
    // std::cout << std::endl << a.get_phi().to_string() << " == " << std::endl;

    a.icrt();
    #ifdef VERBOSE
    std::cout << std::endl << a << " == " << std::endl<< b << std::endl;
    #endif
    BOOST_REQUIRE(a == b);
  }
}


BOOST_AUTO_TEST_CASE(simpleAdd)
{

  Polynomial a;
  Polynomial::random(&a,degree-1);

  Polynomial b(a);//Copy
  Polynomial c = a+b;
  c.icrt();//

  BOOST_REQUIRE(c == a*2);
}

BOOST_AUTO_TEST_CASE(zeroAdd)
{

  Polynomial a;
  Polynomial b;
  Polynomial::random(&a,degree-1);

  Polynomial r = a + b;
  r.icrt();

  #ifdef VERBOSE
  std::cout << "r: " << r << std::endl;
  std::cout << "a: " << a<<std::endl;
  std::cout << "b: " << b<<std::endl;
  #endif
  BOOST_REQUIRE(r == a);

}


BOOST_AUTO_TEST_CASE(multiply)
{
  Polynomial a;
  Polynomial b;

  for(int count = 0; count < NTESTS; count++){

    a.set_device_updated(false);
    b.set_device_updated(false);
    Polynomial::random(&a,degree-1);
    Polynomial::random(&b,degree-1);
    a.set_host_updated(true);
    b.set_host_updated(true);

    ZZ_pEX b_ntl;
    ZZ_pEX a_ntl;
    for(int i = 0;i <= a.deg();i++)
      NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
    for(int i = 0;i <= b.deg();i++)
      NTL::SetCoeff(b_ntl,i,conv<ZZ_p>(b.get_coeff(i)));

    Polynomial c = a*b;
    c.icrt();

    ZZ_pEX c_ntl = a_ntl*b_ntl;

    #ifdef DEBUG
    if(c_ntl != c){
      std::cout << "a: " << a << " degree: " << NTL::deg(a) <<std::endl;
      std::cout << "b: " << b << " degree: " << NTL::deg(b) <<std::endl;
      std::cout << "c: " << c << " degree: " << NTL::deg(c) <<std::endl;
      std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
    }
    std::cout << "count: " << count << std::endl;
    #endif
    BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
    for(int i = 0;i <= c.deg();i++)
      BOOST_REQUIRE(conv<ZZ>(NTL::rep(c_ntl[i])[0]) == c.get_coeff(i));

  }
}

BOOST_AUTO_TEST_CASE(severalMultiplications)
{
  Polynomial a;
  Polynomial b;

  a.set_device_updated(false);
  b.set_device_updated(false);
  Polynomial::random(&a,degree-1);
  Polynomial::random(&b,degree-1);
  a.set_host_updated(true);
  b.set_host_updated(true);

  ZZ_pEX b_ntl;
  ZZ_pEX a_ntl;
  for(int i = 0;i <= a.deg();i++)
    NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
  for(int i = 0;i <= b.deg();i++)
    NTL::SetCoeff(b_ntl,i,conv<ZZ_p>(b.get_coeff(i)));

  Polynomial c = a*b;
  ZZ_pEX c_ntl = a_ntl*b_ntl;

  for(int count = 0; count < NTESTS; count++){
    c *= c;
    c_ntl *= c_ntl;
  }
  c.icrt();

  #ifdef DEBUG
  if(c_ntl != c){
    std::cout << "a: " << a << " degree: " << NTL::deg(a) <<std::endl;
    std::cout << "b: " << b << " degree: " << NTL::deg(b) <<std::endl;
    std::cout << "c: " << c << " degree: " << NTL::deg(c) <<std::endl;
    std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
  }
  std::cout << "count: " << count << std::endl;
  #endif
  BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
  for(int i = 0;i <= c.deg();i++)
    BOOST_REQUIRE(conv<ZZ>(NTL::rep(c_ntl[i])[0]) == c.get_coeff(i));
}

BOOST_AUTO_TEST_CASE(modularInversion)
{
  Polynomial a;
  Polynomial::random(&a,8);


  Polynomial aInv = Polynomial::InvMod(a,a.get_phi());
  Polynomial result = a*aInv % a.get_phi();

  Polynomial one = Polynomial();
  one.set_coeff(0,1);

  BOOST_REQUIRE(result == one);

}

BOOST_AUTO_TEST_CASE(expectedDegree)
{

    Polynomial a;
    Polynomial::random(&a,rand()%degree);

    BOOST_REQUIRE(a.deg() == a.get_expected_degre());

    a.set_coeff(degree+1,1);
    BOOST_REQUIRE(degree+1 == a.get_expected_degre());

}

BOOST_AUTO_TEST_SUITE_END()
