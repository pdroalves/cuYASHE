#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cuYASHE
#include <boost/test/unit_test.hpp>
#include <NTL/ZZ_pEX.h>

#include "polynomial.h"
#include "ciphertext.h"
#include "yashe.h"
#include "common.h"
#include "cuda_functions.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#define NTESTS 100

struct CUDASuite
{
  // Test CUDA functions
    CUDASuite(){

    }

    ~CUDASuite()
    {
        BOOST_TEST_MESSAGE("teardown mass");
    }
};

struct PolySuite
{
  // Test polynomial functions
    int degree;
    Polynomial phi;


    PolySuite(){
        BOOST_TEST_MESSAGE("setup PolySuite");

        degree = 32;
        CUDAFunctions::init(degree);

        Polynomial::global_mod = conv<ZZ>("1171313591017775093490277364417L"); // Defines default GF(q)
        // Polynomial::global_mod = conv<ZZ>("2147483647"); // Defines default GF(q)
        Polynomial::BuildNthCyclotomic(&phi,degree);
        // std::cout << phi.to_string() << std::endl;
        phi.set_mod(Polynomial::global_mod);
        Polynomial::global_phi = &phi;

        srand (36251);

        ZZ_p::init(Polynomial::global_mod);
        ZZ_pX NTL_Phi;
        for(int i = 0; i <= phi.deg();i++){
          NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
        }
        ZZ_pE::init(NTL_Phi);

        Polynomial::gen_crt_primes(Polynomial::global_mod,degree);
    }

    ~PolySuite()
    {
        BOOST_TEST_MESSAGE("teardown mass");
    }
};


struct YasheSuite
{
    ZZ t;
    long t_long;
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
        q = conv<ZZ>("1171313591017775093490277364417L");
        // q = conv<ZZ>("655615111");
        Polynomial::global_mod = q;
        ZZ_p::init(q); // Defines GF(q)

        t_long = 35951;
        // t_long = 2;
        t = ZZ(t_long);
        degree = 4;
        int w = 72;

        Polynomial::BuildNthCyclotomic(&phi, degree); // generate an cyclotomic polynomial
        #ifdef VERBOSE
        std::cout << "Cyclotomic polynomial: " << R << std::endl;
        #endif 
        phi.set_mod(Polynomial::global_mod);
        Polynomial::global_phi = &phi;

        // Set params to NTL (just for comparison reasons)
        ZZ_p::init(Polynomial::global_mod);
        for(int i = 0; i <= phi.deg();i++){
          NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(phi.get_coeff(i)));
        }
        ZZ_pE::init(NTL_Phi);

        CUDAFunctions::init(2*degree);
        
        Polynomial::gen_crt_primes(Polynomial::global_mod,degree);

        // Yashe
        cipher = Yashe();

        Yashe::d = degree;
        Yashe::phi = phi;
        Yashe::q = q;
        // std::cout << ZZ_p::modulus() << std::endl;
        // std::cout << q << std::endl;

        Yashe::t = t;
        Yashe::w = w;
        Yashe::lwq = floor(NTL::log(q)/(log(2)*w)+1);

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

    Polynomial b(a);//Copy
    b.crt();
    b.update_device_data();
    b.set_host_updated(false);

    #ifdef VERBOSE
    std::cout << std::endl << a.to_string() << " == " << std::endl<< b.to_string() << std::endl;
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

  #ifdef VERBOSE
  std::cout << "a: " << a.to_string() << std::endl;
  std::cout << "b: " << b.to_string() << std::endl;
  std::cout << "c: " << c.to_string() << std::endl;
  #endif
  BOOST_REQUIRE(c == (a*2)%(c.get_mod()));
}

BOOST_AUTO_TEST_CASE(multipleAdds)
{

  Polynomial a;
  Polynomial::random(&a,degree-1);

  Polynomial b;
  for(int count = 0; count < NTESTS;count ++)
    b += a;

  #ifdef VERBOSE
  std::cout << "a: " << a.to_string() << std::endl;
  std::cout << "b: " << b.to_string() << std::endl;
  #endif
  BOOST_REQUIRE(b == (a*NTESTS)%(b.get_mod()));
}

BOOST_AUTO_TEST_CASE(multipleAddsWithDifferentDegrees)
{

  Polynomial b;
  ZZ_pEX b_ntl;

  for(int count = 0; count < NTESTS;count ++){
    Polynomial a;
    Polynomial::random(&a,rand() % (degree-1));

    ZZ_pEX a_ntl;
    for(int i = 0;i <= a.deg();i++)
      NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));

    b += a;
    b_ntl += a_ntl;
  }

  BOOST_REQUIRE(NTL::deg(b_ntl) == b.deg());
  for(int i = 0;i <= b.deg();i++)
    BOOST_REQUIRE(conv<ZZ>(NTL::rep(b_ntl[i])[0]) == b.get_coeff(i));
}

BOOST_AUTO_TEST_CASE(zeroAdd)
{

  Polynomial a;
  Polynomial b;
  Polynomial::random(&a,degree-1);

  Polynomial r = a + b;
  // r.icrt();

  #ifdef VERBOSE
  std::cout << "r: " << r << std::endl;
  std::cout << "a: " << a<<std::endl;
  std::cout << "b: " << b<<std::endl;
  #endif
  BOOST_REQUIRE(r == a);

}

BOOST_AUTO_TEST_CASE(wNTest)
{
  cuyasheint_t *W;
  cuyasheint_t *WInv;
  int N = degree;

  W = (cuyasheint_t*)malloc(N*N*sizeof(cuyasheint_t));
  WInv = (cuyasheint_t*)malloc(N*N*sizeof(cuyasheint_t));

  cudaError_t result = cudaMemcpy(W,CUDAFunctions::d_W , N*N*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
  BOOST_REQUIRE(result == cudaSuccess);
  result = cudaMemcpy(WInv,CUDAFunctions::d_WInv , N*N*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
  BOOST_REQUIRE(result == cudaSuccess);

  // const cuyasheint_t P = 0xffffffff00000001;
  cuyasheint_t wN = CUDAFunctions::wN;

  ZZ PZZ = conv<ZZ>("2147483647");
  cuyasheint_t P = 2147483647;
  cuyasheint_t k = conv<cuyasheint_t>(PZZ-1)/N;
  ZZ wNZZ = NTL::PowerMod(ZZ(7),k,PZZ);

  BOOST_REQUIRE(W[1+N] == wN);
  BOOST_REQUIRE(wNZZ == wN);
  BOOST_REQUIRE(W[2+N] == conv<cuyasheint_t>(NTL::PowerMod(wNZZ,2,PZZ)));
  BOOST_REQUIRE(W[2+2*N] == conv<cuyasheint_t>(NTL::PowerMod(wNZZ,4,PZZ)));

}

BOOST_AUTO_TEST_CASE(sremTest)
{
  for(int i = 0; i < 100*NTESTS;i++){
    uint64_t a = rand()*pow(2,31);
    cuyasheint_t result = s_rem(a);
    // std::cout << "a: " << a << std::endl << "result: " << result << " == " << (a%2147483647)<<std::endl;
    BOOST_REQUIRE(result == (a%2147483647));
  }

}
BOOST_AUTO_TEST_CASE(simpleMultiplication)
{
   // std:: cout <<  NTL::MulMod(6406262673276882058,4,9223372036854829057) << std::endl;
  // std:: cout <<  NTL::MulMod(6406262673276882058,4,9223372036854829057) << std::endl;

// cuyasheint_t integer = 9223372036854829057L;
//     ZZ P = ZZ(integer);
//     ZZ x = Z(6209464568650184525);
//     ZZ inv = NTL::InvMod(x,P);
    
//     cout << integer << "\n" << P << endl;
    
//     cout << "PowerMod: " << inv << endl;
    
//     cout << "Check: " << NTL::MulMod(inv, Z(6209464568650184525), P) << endl;    
//   return 0;
  for(int N = degree;N <= 2048;N *= 2){
    CUDAFunctions::init(N);

    const int NPOLYS = Polynomial::CRTPrimes.size();

    dim3 blockDim(ADDBLOCKXDIM);
    dim3 gridDim((N*NPOLYS)/ADDBLOCKXDIM+1);

    cuyasheint_t *h_a;
    cuyasheint_t *d_a;
    cuyasheint_t *h_b;
    cuyasheint_t *d_b;
    cuyasheint_t *h_c;

    // Alloc memory
    h_a = (cuyasheint_t*)malloc(N*NPOLYS*sizeof(cuyasheint_t));
    h_b = (cuyasheint_t*)malloc(N*NPOLYS*sizeof(cuyasheint_t));
    h_c = (cuyasheint_t*)malloc(N*NPOLYS*sizeof(cuyasheint_t));
    cudaError_t result = cudaMalloc((void**)&d_a,N*NPOLYS*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_b,N*NPOLYS*sizeof(cuyasheint_t));
    assert(result == cudaSuccess);

    // Generates random values

    ZZ_pEX a_ntl;
    ZZ_pEX b_ntl;
    for(int j = 0; j < NPOLYS;j++)
      for(int i = 0; i < N; i++){
        if(i < N/2){
          h_a[i+j*N] = i;        
          NTL::SetCoeff(a_ntl,i,i);
          h_b[i+j*N] = 1;
          NTL::SetCoeff(b_ntl,i,1);        
        }else{        
          h_a[i+j*N] = 0;
          NTL::SetCoeff(a_ntl,i,0);
          h_b[i+j*N] = 0;
          NTL::SetCoeff(b_ntl,i,0);       
        }
      }

    // Copy to GPU
    result = cudaMemcpy(d_a,h_a , N*NPOLYS*sizeof(cuyasheint_t), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    // result = cudaMemset((void*)d_b,1,N*NPOLYS*sizeof(cuyasheint_t));
    // assert(result == cudaSuccess);  
    result = cudaMemcpy(d_b,h_b , N*NPOLYS*sizeof(cuyasheint_t), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    // Multiply
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cuyasheint_t *d_c = CUDAFunctions::callPolynomialMul(stream,d_a,d_b,N, NPOLYS);

    ZZ_pEX c_ntl = a_ntl*b_ntl;

    result = cudaMemcpy(h_c,d_c,  N*NPOLYS*sizeof(cuyasheint_t), cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);
    // std::cout << c_ntl << std::endl;
    for(int j = 0; j < NPOLYS;j++)
      for(int i = 0; i < N; i++){
        ZZ ntl_value;
        if( NTL::IsZero(NTL::coeff(c_ntl,i)) )
        // Without this, NTL raises an exception when we call rep()
          ntl_value = 0L;
        else
          ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(c_ntl,i))[0]);

        BOOST_REQUIRE(h_c[i+j*N] == ntl_value);
      }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
  }
}

BOOST_AUTO_TEST_CASE(multiplyByZZOnCPU)
{
  Polynomial a;
  ZZ b;

  for(int count = 0; count < NTESTS; count++){

    a.set_device_updated(false);
    Polynomial::random(&a,degree-1);
    a.set_host_updated(true);

    ZZ_pEX a_ntl;
    for(int i = 0;i <= a.deg();i++)
      NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
    NTL::RandomBnd(b,ZZ(42));
    // b = ZZ(0);

    Polynomial c = a*b % Polynomial::global_mod;
    c.icrt();
    c.normalize();

    ZZ_pEX c_ntl = a_ntl*conv<ZZ_p>(b);

    #ifdef VERBOSE
    std::cout << "a: " << a.to_string() << " degree: " << a.deg() <<std::endl;
    std::cout << "b: " << b <<std::endl;
    std::cout << "c: " << c.to_string() << " degree: " << c.deg() <<std::endl;
    std::cout << ": " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
    std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
    std::cout << "count: " << count << std::endl;
    #endif
    BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
    for(int i = 0;i <= c.deg();i++)
      BOOST_REQUIRE(conv<ZZ>(NTL::rep(c_ntl[i])[0]) == c.get_coeff(i));
  }
}

// BOOST_AUTO_TEST_CASE(multiplyByZZOnGPU)
// {
//   Polynomial a;
//   ZZ b;

//   for(int count = 0; count < NTESTS; count++){

//     a.set_device_updated(false);
//     Polynomial::random(&a,degree-1);
//     a.set_host_updated(true);

//     ZZ_pEX a_ntl;
//     for(int i = 0;i <= a.deg();i++)
//       NTL::SetCoeff(a_ntl,i,conv<ZZ_p>(a.get_coeff(i)));
//     NTL::RandomBnd(b,ZZ(42));
//     // b = ZZ(0);

//     // This will force the execution on GPU
//     a.crt();
//     a.update_device_data();
//     a.set_host_updated(false);

//     Polynomial c = a*b % Polynomial::global_mod;
//     c.icrt();
//     c.normalize();

//     ZZ_pEX c_ntl = a_ntl*conv<ZZ_p>(b);

//     #ifdef VERBOSE
//     std::cout << "a: " << a.to_string() << " degree: " << a.deg() <<std::endl;
//     std::cout << "b: " << b <<std::endl;
//     std::cout << "c: " << c.to_string() << " degree: " << c.deg() <<std::endl;
//     std::cout << ": " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
//     std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
//     std::cout << "count: " << count << std::endl;
//     #endif
//     BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
//     for(int i = 0;i <= c.deg();i++)
//       BOOST_REQUIRE(conv<ZZ>(NTL::rep(c_ntl[i])[0]) == c.get_coeff(i));
//   }
// }

BOOST_AUTO_TEST_CASE(simpleMultiplyByPolynomial)
{
  Polynomial a;
  Polynomial b;

  CUDAFunctions::init(2*degree);


  a.set_device_updated(false);
  b.set_device_updated(false);
  for(int i = 0; i < degree;i++){
    a.set_coeff(i,i);
    b.set_coeff(i,1);
  }
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

BOOST_AUTO_TEST_CASE(multiplyByPolynomial)
{

  CUDAFunctions::init(2*degree);

  for(int count = 0; count < NTESTS; count++){

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
    c %= Polynomial::global_mod;

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

  CUDAFunctions::init(2*degree);

  for(int count = 0; count < NTESTS; count++){

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

    Polynomial c = a*b+a;

    ZZ_pEX c_ntl = a_ntl*b_ntl+a_ntl;

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

  CUDAFunctions::init(2*degree);

  for(int count = 0; count < NTESTS; count++){

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

    Polynomial c = a*b+a;

    ZZ_pEX c_ntl = a_ntl*b_ntl+a_ntl;

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

BOOST_AUTO_TEST_CASE(randomPolynomialOperations)
{

  CUDAFunctions::init(2*degree);


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
  Polynomial c;
  ZZ_pEX c_ntl;

  for(int count = 0; count < 10*NTESTS; count++){
    int random_op_bit = rand()%2;
    int random_ab = rand()%3;
    // 0: add
    if(random_op_bit){
      // 1: mul
      if(random_ab == 1){
        // 1: b
        c *= b;
        c_ntl *= b_ntl;
      }else if(random_ab == 0){
        // 0: a
        c *= a;
        c_ntl *= a_ntl;
      }else if(random_ab == 2){
        // 2: some integer
        long value = conv<long>(ZZ(rand()) % c.get_mod());
        c *= value;
        c_ntl *= value;
      }
    }else{
      // 0: add
      if(random_ab == 1){
        // 1: b
        c += b;
        c_ntl += b_ntl;
      }else if(random_ab == 0){
        // 0: a
        c += a;
        c_ntl += a_ntl;
      }else if(random_ab == 2){
        // 2: some integer
        long value = conv<long>(ZZ(rand()) % c.get_mod());
        c += value;
        c_ntl += value;
      }

    }

    c %= phi;
    c %= Polynomial::global_mod;
    c.normalize();
    c_ntl %= conv<ZZ_pEX>(ZZ_pE::modulus());



  // #ifdef DEBUG
  // std::cout << "c: " << c.to_string() << " degree: " << c.deg() <<std::endl;
  // std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;
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

    // std::cout << "c: " << c.to_string() << " degree: " << c.deg() <<std::endl;
    // std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;

  
}

BOOST_AUTO_TEST_CASE(modularInversion)
{

  CUDAFunctions::init(2*degree);

  Polynomial a;
  Polynomial::random(&a,degree-1);


  Polynomial aInv = Polynomial::InvMod(a,a.get_phi());
  Polynomial result = a*aInv % a.get_phi();
  result %= a.get_mod();

  Polynomial one = Polynomial();
  one.set_coeff(0,1);

  result.normalize();

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

BOOST_FIXTURE_TEST_SUITE(CUDAFixture, CUDASuite)

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(YasheFixture, YasheSuite)

BOOST_AUTO_TEST_CASE(encryptDecrypt)
{
  Polynomial a;

  for(int i = 0; i < 10*NTESTS;i++){
    a.set_coeff(0,conv<ZZ>(rand())%t);

    Ciphertext c = cipher.encrypt(a);
    Polynomial a_decrypted = cipher.decrypt(c);

    BOOST_REQUIRE( a_decrypted == a);
  }
}

BOOST_AUTO_TEST_SUITE_END()
