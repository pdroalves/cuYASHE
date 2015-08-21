#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cuYASHE
#include <boost/test/unit_test.hpp>
#include <NTL/ZZ_pEX.h>

#include "polynomial.h"
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

        degree = 128;
        CUDAFunctions::init(degree);

        Polynomial::global_mod = conv<ZZ>("1171313591017775093490277364417"); // Defines default GF(q)
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
  // c.icrt();//

  #ifdef VERBOSE
  std::cout << "a: " << a.to_string() << std::endl;
  std::cout << "b: " << b.to_string() << std::endl;
  std::cout << "c: " << c.to_string() << std::endl;
  #endif
  BOOST_REQUIRE(c == a*2);
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
  BOOST_REQUIRE(b == a*NTESTS);
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
  uint64_t *W;
  uint64_t *WInv;
  int N = degree;

  W = (uint64_t*)malloc(N*N*sizeof(uint64_t));
  WInv = (uint64_t*)malloc(N*N*sizeof(uint64_t));

  cudaError_t result = cudaMemcpy(W,CUDAFunctions::d_W , N*N*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  BOOST_REQUIRE(result == cudaSuccess);
  result = cudaMemcpy(WInv,CUDAFunctions::d_WInv , N*N*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  BOOST_REQUIRE(result == cudaSuccess);

  const uint64_t P = 0xffffffff00000001;
  uint64_t wN = CUDAFunctions::wN;

  BOOST_REQUIRE(W[1+N] == wN);
  BOOST_REQUIRE(W[2+N] == NTL::PowerMod(wN,2,P));
  BOOST_REQUIRE(W[2+2*N] == NTL::PowerMod(wN,4,P));

  BOOST_REQUIRE(NTL::MulMod(W[1+N],WInv[1+N],P) == 1);
  BOOST_REQUIRE(NTL::MulMod(W[2+N],WInv[2+N],P) == 1);
  BOOST_REQUIRE(NTL::MulMod(W[2+2*N],WInv[2+2*N],P) == 1);
}

BOOST_AUTO_TEST_CASE(simpleMultiplication)
{
   // std:: cout <<  NTL::MulMod(6406262673276882058,4,9223372036854829057) << std::endl;
  // std:: cout <<  NTL::MulMod(6406262673276882058,4,9223372036854829057) << std::endl;

// uint64_t integer = 9223372036854829057L;
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

    uint64_t *h_a;
    uint64_t *d_a;
    uint64_t *h_b;
    uint64_t *d_b;
    uint64_t *h_c;

    // Alloc memory
    h_a = (uint64_t*)malloc(N*NPOLYS*sizeof(uint64_t));
    h_b = (uint64_t*)malloc(N*NPOLYS*sizeof(uint64_t));
    h_c = (uint64_t*)malloc(N*NPOLYS*sizeof(uint64_t));
    cudaError_t result = cudaMalloc((void**)&d_a,N*NPOLYS*sizeof(uint64_t));
    assert(result == cudaSuccess);
    result = cudaMalloc((void**)&d_b,N*NPOLYS*sizeof(uint64_t));
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
    result = cudaMemcpy(d_a,h_a , N*NPOLYS*sizeof(uint64_t), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    // result = cudaMemset((void*)d_b,1,N*NPOLYS*sizeof(uint64_t));
    // assert(result == cudaSuccess);  
    result = cudaMemcpy(d_b,h_b , N*NPOLYS*sizeof(uint64_t), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    // Multiply
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    uint64_t *d_c = CUDAFunctions::callPolynomialMul(stream,d_a,d_b,N, NPOLYS);

    ZZ_pEX c_ntl = a_ntl*b_ntl;

    result = cudaMemcpy(h_c,d_c,  N*NPOLYS*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);
    // std::cout << c_ntl << std::endl;
    for(int j = 0; j < NPOLYS;j++)
      for(int i = 0; i < N; i++){
        int ntl_value;
        if( NTL::IsZero(NTL::coeff(c_ntl,i)) )
        // Without this, NTL raises an exception when we call rep()
          ntl_value = 0L;
        else
          ntl_value = conv<int>(NTL::rep(NTL::coeff(c_ntl,i))[0]);

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

BOOST_AUTO_TEST_CASE(multiplyByZZOnGPU)
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

    // This will force the execution on GPU
    a.crt();
    a.update_device_data();
    a.set_host_updated(false);

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



BOOST_AUTO_TEST_CASE(multiplyByPolynomial)
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
    std::cout << "c: " << c.to_string() << " degree: " << c.deg() << std::endl << std::endl;
    std::cout << "c_ntl: " << c_ntl << " degree: " << NTL::deg(c_ntl) << std::endl << std::endl;

    BOOST_REQUIRE(NTL::deg(c_ntl) == c.deg());
    for(int i = 0;i <= c.deg();i++)
      BOOST_REQUIRE(conv<ZZ>(NTL::rep(c_ntl[i])[0]) == c.get_coeff(i));

  }
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

BOOST_FIXTURE_TEST_SUITE(CUDAFixture, CUDASuite)

BOOST_AUTO_TEST_SUITE_END()
