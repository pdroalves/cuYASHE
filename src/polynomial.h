#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H
#include <stdio.h>
#include <vector>
#include <sstream>
#include <NTL/ZZ.h>
#include <cuda_runtime.h>
#include "cuda_functions.h"
#include "common.h"
#include <algorithm>

NTL_CLIENT

class Polynomial{
  public:
    // Attributes
    int CRTSPACING =-1;// Stores the distance between the zero-coeff of two consecutive residues in d_polyCRT
    static ZZ CRTProduct;
    static std::vector<long> CRTPrimes;
    static Polynomial *global_phi;
    static ZZ global_mod;

    // Constructors
    Polynomial(){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      if(this->global_mod > 0){
        // If a global mod is defined, use it
        this->mod = this->global_mod; // Doesn't copy. Uses the reference.
      }
      if(this->global_phi){
        // If a global phi is defined, use it
        this->phi = this->global_phi; // Doesn't copy. Uses the reference.

        // If the irreductible polynomial have degree N, this polynomial's degree will be limited to N-1
        this->CRTSPACING = this->global_phi->deg();
      }else{
        // CRT Spacing not set
        this->CRTSPACING = -1;
      }
      coefs.push_back(ZZ(0));

      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Polynomial(ZZ p){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->mod = ZZ(p);// Copy

      if(this->global_phi){
      //   // If a global phi is defined, use it
        this->phi = this->global_phi; // Doesn't copy. Uses the reference.
      }
      // CRT Spacing not set
      this->CRTSPACING = -1;

      coefs.push_back(ZZ(0));
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod << " but no phi."<< std::endl;
      #endif
    }
    Polynomial(ZZ p,Polynomial P){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->mod = ZZ(p);// Copy
      *(this->phi) = Polynomial(P);// Copy

      // CRT Spacing should store the expected number of coefficients
      // If the irreductible polynomial have degree N, this polynomial's degree will be limited to N-1
      this->CRTSPACING = this->phi->deg();

      coefs.push_back(ZZ(0));
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this-> phi << std::endl;
      #endif
    }
    Polynomial(int spacing){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      if(this->global_mod > 0){
        // If a global mod is defined, use it
        this->mod = this->global_mod; // Doesn't copy. Uses the reference.
      }
      if(this->global_phi){
      //   // If a global phi is defined, use it
        this->phi = this->global_phi; // Doesn't copy. Uses the reference.
      }
      // CRT Spacing set to spacing
      this->CRTSPACING = spacing;
      coefs.push_back(ZZ(0));

      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << " and no mod or phi elements" << std::endl;
      #endif
    }
    Polynomial(ZZ p,Polynomial P,int spacing){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->mod = ZZ(p);// Copy
      this->phi = &P;// Copy

      // std::cout << this->get_phi().to_string() << std::endl;

      // CRT Spacing set to spacing
      this->CRTSPACING = spacing;

      coefs.push_back(ZZ(0));
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod <<" and phi " << this->phi << std::endl;
      #endif
    }
    Polynomial(ZZ p,int spacing){
      cudaStreamCreate(&this->stream);
      this->expected_degree = -1;
      this->set_host_updated(true);
      this->set_device_updated(false);
      this->mod = ZZ(p);// Copy

      // CRT Spacing set to spacing
      this->CRTSPACING = spacing;

      coefs.push_back(ZZ(0));
      #ifdef VERBOSE
        std::cout << "Polynomial constructed with CRTSPACING " << this->CRTSPACING << ", mod "  << this->mod << std::endl;
      #endif
    }
    Polynomial(Polynomial *b){
      // Copy
      this->copy(*b);
    }

    void copy(Polynomial b){
      this->CRTSPACING = b.CRTSPACING;
      this->set_host_updated(b.get_host_updated());
      this->set_device_updated(b.get_device_updated());
      this->set_coeffs(b.get_coeffs());
      this->polyCRT = b.get_crt_residues();
      this->d_polyCRT = b.get_device_crt_residues();
      this->set_mod(b.get_mod());
      this->expected_degree = b.deg();
      if(this != Polynomial::global_phi)
        this->set_phi(b.get_phi());
    }
    // Functions and methods
    // Operators
    // std::ostream &operator<<(std::ostream &os, Polynomial &m) {
    //   for(int i = 0; i <=  m.deg();i++)
    //     os << m.get_coeff(i);
    //   return os;
    // }

    std::string to_string(){
      if(!this->get_host_updated()){
        this->icrt();
      }

      stringstream ss;
      for(int i = 0; i <=  this->deg();i++)
        ss << this->get_coeff(i) << ", ";
      return ss.str();;
    }
    Polynomial operator=(Polynomial b){//Copy
        this->copy(&b);

        #ifdef VERBOSE
          std::cout << "Polynomial copied. " << std::endl;
        #endif
        return *this;
    }
    Polynomial operator+(Polynomial b){
      // Check align
      if(this->CRTSPACING != b.CRTSPACING){
        int new_spacing = std::max(this->CRTSPACING,b.CRTSPACING);
        this->update_crt_spacing(new_spacing);
        b.update_crt_spacing(new_spacing);
      }

      #ifdef VERBOSE
      std::cout << "Adding:" << std::endl;
      // std::cout << "this: " << this->to_string() << std::endl;
      // std::cout << "other " << b.to_string() << std::endl;
      #endif

      // Apply CRT and copy data to global memory, if needed
      #pragma omp parallel sections num_threads(2)
      {
          #pragma omp section
          {

              if(!this->get_device_updated()){
                this->crt();
                this->update_device_data();
              }

          }
          #pragma omp section
          {
              if(!b.get_device_updated()){
                  b.crt();
                  b.update_device_data();
              }
          }
      }


      long *d_result = CUDAFunctions::callPolynomialAddSub(this->stream,this->get_device_crt_residues(),b.get_device_crt_residues(),(int)(this->CRTSPACING*this->polyCRT.size()),ADD);

      Polynomial c(this->get_mod(),this->get_phi(),this->CRTSPACING);
      c.set_device_crt_residues(d_result);
      c.set_host_updated(false);
      c.set_device_updated(true);
      cudaDeviceSynchronize();
      return c;
    }
    Polynomial operator+=(Polynomial b){
      this->copy( ((*this)+b));
      return *this;
    }
    Polynomial operator-(Polynomial b){
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
      #pragma omp parallel sections
      {
          #pragma omp section
          {

              if(!this->get_device_updated()){
                this->crt();
                this->update_device_data();
              }

          }
          #pragma omp section
          {
              if(!b.get_device_updated()){
                  b.crt();
                  b.update_device_data();
              }
          }
      }

      long *d_result = CUDAFunctions::callPolynomialAddSub(this->stream,this->get_device_crt_residues(),b.get_device_crt_residues(),(int)(this->CRTSPACING*this->polyCRT.size()),SUB);

      Polynomial c(this->get_mod(),this->get_phi(),this->CRTSPACING);
      c.set_device_crt_residues(d_result);
      c.set_host_updated(false);
      c.set_device_updated(true);
      cudaDeviceSynchronize();
      return c;
    }
    Polynomial operator-=(Polynomial b){
      this->copy( ((*this)-b));
      return *this;
    }
    Polynomial operator*(Polynomial b){

      // Check align
      if(this->CRTSPACING != b.CRTSPACING){
        int new_spacing = std::max(this->CRTSPACING,b.CRTSPACING);
        this->update_crt_spacing(new_spacing);
        b.update_crt_spacing(new_spacing);
      }

      #ifdef VERBOSE
      std::cout << "Mul:" << std::endl;
      // std::cout << "this: " << this->to_string() << std::endl;
      // std::cout << "other " << b.to_string() << std::endl;
      #endif

      // Apply CRT and copy data to global memory, if needed
      #pragma omp parallel sections
      {
          #pragma omp section
          {

              if(!this->get_device_updated()){
                this->crt();
                this->update_device_data();
              }

          }
          #pragma omp section
          {
              if(!b.get_device_updated()){
                  b.crt();
                  b.update_device_data();
              }
          }
      }

      long *d_result = CUDAFunctions::callPolynomialMul(this->stream,this->get_device_crt_residues(),b.get_device_crt_residues(),this->CRTSPACING,this->CRTPrimes.size());

      Polynomial c(this->get_mod(),this->get_phi(),2*this->CRTSPACING);
      c.set_device_crt_residues(d_result);
      c.set_host_updated(false);
      c.set_device_updated(true);
      cudaDeviceSynchronize();
      return c;
    }
    Polynomial operator*=(Polynomial b){
      this->copy( ((*this)*b));
      return *this;
    }
    Polynomial operator/(Polynomial b){
      Polynomial quot;
      Polynomial rem;

      #pragma omp parallel sections
      {
          #pragma omp section
          {
            if(!this->get_host_updated())
              this->icrt();
          }
          #pragma omp section
          {
            if(!b.get_host_updated())
              b.icrt();
          }
      }

      Polynomial::DivRem((*this),b,quot, rem);
      return quot;
    }
    Polynomial operator/=(Polynomial b){
      this->copy( ((*this)/b));
      return *this;
    }
    Polynomial operator%(Polynomial b){
      Polynomial quot;
      Polynomial rem;

      #pragma omp parallel sections
      {
          #pragma omp section
          {
            if(!this->get_host_updated())
              this->icrt();
          }
          #pragma omp section
          {
            if(!b.get_host_updated())
              b.icrt();
          }
      }

      Polynomial::DivRem((*this),b,quot, rem);
      // rem.icrt();
      // std::cout << rem.to_string() << std::endl;
      this->copy(&rem);
      return *this;
    }
    Polynomial operator%=(Polynomial b){
      this->copy( ((*this)%b));
      return *this;
    }

    Polynomial operator+(ZZ b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        // Convert to polynomial and send to device
        Polynomial B(this->get_mod(),this->get_phi(),this->get_crt_spacing());
        B.set_coeff(0,b);
        return p+B;
      }

      p.set_coeff(0,p.get_coeff(0)+b);
      p.set_device_updated(false);
      return p;
    }
    Polynomial operator+=(ZZ b){
      this->set_coeffs( ((*this)+b).get_coeffs());
      return *this;
    }
    Polynomial operator-(ZZ b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        // Convert to polynomial and send to device
        Polynomial B(this->get_mod(),this->get_phi(),this->get_crt_spacing());
        B.set_coeff(0,b);
        return p-B;
      }
      p.set_coeff(0,p.get_coeff(0)-b);
      p.set_device_updated(false);
      return p;
    }
    Polynomial operator-=(ZZ b){
      this->set_coeffs( ((*this)-b).get_coeffs());
      return *this;
    }
    Polynomial operator*(ZZ b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        // Convert to polynomial and send to device
        Polynomial B(this->get_mod(),this->get_phi(),this->get_crt_spacing());
        B.set_coeff(0,b);
        return p*B;
      }

      #pragma omp parallel for
      for(int i = 0; i <= p.deg();i++)
        p.set_coeff(i,conv<ZZ>(p.get_coeff(i)*b));
      p.set_device_updated(false);
      return p;
    }
    Polynomial operator*=(ZZ b){
      this->set_coeffs( ((*this)*b).get_coeffs());
      return *this;
    }
    Polynomial operator%(ZZ b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        throw "Polynomial mod on device not implemented!";
      }

      #pragma omp parallel for
      for(int i = 0; i <= p.deg();i++){
        // ZZ value = p.get_coeff(i)%b;
        // std::cout << "value: " << value << std::endl << "b: " << b<< std::endl;
        // p.set_coeff(i,value);
        p.set_coeff(i,p.get_coeff(i)%b);
      }
      p.set_device_updated(false);

      return p;
    }
    Polynomial operator%=(ZZ b){
      this->set_coeffs( ((*this)%b).get_coeffs());
      return *this;
    }
    Polynomial operator/(ZZ b){
      Polynomial p(*this);
      if(!this->get_host_updated()){
        throw "Polynomial division on device not implemented!";
      }

      #pragma omp parallel for
      for(int i = 0; i <= p.deg();i++)
        p.set_coeff(i,p.get_coeff(i)/b);
      p.set_device_updated(false);

      return p;
    }
    Polynomial operator/=(ZZ b){
      this->set_coeffs( ((*this)/b).get_coeffs());
      return *this;
    }
    Polynomial operator+(int b){
      return (*this)+ZZ(b);
    }
    Polynomial operator+=(int b){
      this->set_coeffs( ((*this)+b).get_coeffs());
      return *this;
    }
    Polynomial operator-(int b){
      return (*this)-ZZ(b);
    }
    Polynomial operator-=(int b){
      this->set_coeffs( ((*this)-b).get_coeffs());
      return *this;
    }
    Polynomial operator*(int b){
      return (*this)*ZZ(b);
    }
    Polynomial operator*=(int b){
      this->set_coeffs( ((*this)*conv<ZZ>(b)).get_coeffs());
      return *this;
    }
    Polynomial operator/(int b){
      return (*this)/ZZ(b);
    }
    Polynomial operator/=(int b){
      this->set_coeffs( ((*this)/conv<ZZ>(b)).get_coeffs());
      return *this;
    }

  bool operator==(Polynomial b){
      if(!this->get_host_updated()){
          this->icrt();
      }
      if(!b.get_host_updated()){
        b.icrt();
      }

      if(this->deg() != b.deg())
        return false;

      for(unsigned int i = 0; i <= this->deg();i++){
        if(this->get_coeff(i) != b.get_coeff(i))
          return false;
      }
      return true;
    }
    bool operator!=(Polynomial b){
      return !((*this) == b);
    }

    void CPUAddition(Polynomial *b){
      // Forces the addition to be executed by CPU
      // This method supposes that there is no need to apply CRT/ICRT on operands
      for(unsigned int i = 0; i <= std::max(this->deg(),b->deg()); i++)
        this->set_coeff(i,this->get_coeff(i) + b->get_coeff(i));
    }

    void CPUSubtraction(Polynomial *b){
      // Forces the subtraction to be executed by CPU
      // This method supposes that there is no need to apply CRT/ICRT on operands
      for(unsigned int i = 0; i <= b->deg(); i++)
        this->set_coeff(i,this->get_coeff(i) - b->get_coeff(i));
    }

    void normalize(){
      if(!this->get_host_updated()){
        this->icrt();
      }

      // Remove last 0-coefficients
      while(this->deg() >= 0 &&
            this->get_coeff(this->deg()) == ZZ(0))
        this->coefs.pop_back();

    }
    // gets and sets
    ZZ get_mod(){
      // Returns a copy of mod
      return ZZ(this->mod);
    }
    void set_mod(ZZ value){
      this->mod = ZZ(value);
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
    }


    ZZ get_coeff(const int index){

      if(!this->get_host_updated()){
        this->icrt();
      }

      // Returns a copy of coefficient at this index
      if(index > this->deg())
        return conv<ZZ>(0);
      else
        return this->coefs.at(index);
    }
    void set_coeff(int index,ZZ value){

      if(!this->get_host_updated()){
        this->icrt();
      }

      if((unsigned int)(index) >= this->coefs.size()){
        #ifdef VERBOSE
          std::cout << "Resizing from "<< this->coefs.size() << " to " << index+1 << std::endl;
        #endif

        this->coefs.resize(index+1);
      }
      this->coefs[index] = value;

      this->expected_degree = this->coefs.size()-1;
      #ifdef VERBOSE
        std::cout << "Polynomial coeff " << index << " set to " << this->coefs[index] << std::endl;
      #endif
    }
    void set_coeff(int index,int value){

      this->set_coeff(index,ZZ(value));
    }
    std::vector<ZZ> get_coeffs(){

      if(!this->get_host_updated()){
        this->icrt();
      }

      // Returns a copy of all coefficients
      std::vector<ZZ> coefs_copy(this->coefs);
      return coefs_copy;
    }
    void set_coeffs(std::vector<long> values){

      if(!this->get_host_updated()){
        this->icrt();
      }

      // Replaces all coefficients
      this->coefs.resize(values.size());
      for(std::vector<long>::iterator iter = values.begin();iter != values.end();iter++){
        this->coefs[iter-values.begin()] = conv<ZZ>(*iter);
      }
      this->expected_degree = this->coefs.size()-1;

    }
    void set_coeffs(std::vector<ZZ> values){

      if(!this->get_host_updated()){
        this->icrt();
      }

      // Replaces all coefficients
      this->coefs = values;
      this->expected_degree = this->coefs.size()-1;

    }
    std::vector<std::vector<long> > get_crt_residues(){
      std::vector<std::vector<long> > crt_residues_copy(this->polyCRT);
      return crt_residues_copy;
    }
    long* get_device_crt_residues(){
      // Returns the address of crt residues at device memory
      return this->d_polyCRT;
    }
    void set_device_crt_residues(long *residues){
      this->d_polyCRT = residues;
    }

    void crt();
    void icrt();
    int get_crt_spacing(){
      return this->CRTSPACING;
    }
    static void gen_crt_primes(ZZ q,long degree){
        // We will use 63bit primes to fit long data type (64 bits raises "GenPrime: length too large")
        ZZ M = ZZ(1);
        std::vector<long> P;

        int primes_size = CRTPRIMESIZE;
        long n;

        while( (M < (2*degree)*q*q*q) ){
            n = NTL::GenPrime_long(primes_size);
            if( std::find(P.begin(), P.end(), n) == P.end()){
              // Does not contains
              P.push_back(n);
              M *=(n);
            }
        }

        Polynomial::CRTProduct = M;
        Polynomial::CRTPrimes = P;

        std::cout << P.size() << " primes generated." << std::endl;
        #ifdef DEBUG
        std::cout << "Primes set - M:" << Polynomial::CRTProduct << std::endl;
        // std::cout << "Primes: "<< Polynomial::CRTPrimes << std::endl;
        #endif
    }

    void update_device_data();
    void set_device_updated(bool b){
      this->DEVICE_IS_UPDATE = b;
    }
    bool get_device_updated(){
      return this->DEVICE_IS_UPDATE;
    }
    void update_host_data();
    void set_host_updated(bool b){
      this->HOST_IS_UPDATED = b;
    }
    bool get_host_updated(){
      return this->HOST_IS_UPDATED;
    }


    int deg(){
      return this->coefs.size()-1;
    }
    ZZ lead_coeff(){
      if(this->deg() >= 0){
        return this->get_coeff(this->deg());
      }else{
        return ZZ(0);
      }
    }
    static bool check_special_rem_format(Polynomial p){
      if(p.get_coeff(0) != 1){
        return false;
      }
      if(p.lead_coeff() != ZZ(1)){
        return false;
      }
      for(int i =1;i < p.deg();i++){
        if(p.get_coeff(i) != 0)
          return false;
      }
      return true;
    }
    void update_crt_spacing(int new_spacing){
      if(this->CRTSPACING == new_spacing)
        return;

      // If updated data lies in gpu's global memory, realign it
      if(this->get_device_updated()){
        long * d_pointer = CUDAFunctions::callRealignCRTResidues(this->stream, this->CRTSPACING,new_spacing,this->get_device_crt_residues(),this->deg()+1,Polynomial::CRTPrimes.size());
        if(d_pointer != NULL){
          this->set_device_crt_residues(d_pointer);
        }else{
          #ifdef VERBOSE
          std::cout << "Old spacing is equal new spacing." << std::endl;
          #endif
        }
      }
      this->CRTSPACING = new_spacing;

    }
    int get_expected_degre(){
      return expected_degree;
    }
    static void DivRem(Polynomial a,Polynomial b,Polynomial &quot,Polynomial &rem);
    static Polynomial InvMod(Polynomial a,Polynomial b){
      // To-do
      throw "Polynomial InvMod not implemented!";
    }
    static void BuildNthCyclotomic(Polynomial *phi, int n);
    static void random(Polynomial *a,int degree){
      if(a->get_mod() > 0)
        for(int i = 0;i <= degree;i++)
          a->set_coeff(i,ZZ(rand())%a->get_mod());
      else
        for(int i = 0;i <= degree;i++)
          a->set_coeff(i,ZZ(rand()) % a->global_mod);
    }
  private:
    // Attributes
    cudaStream_t stream;
    int expected_degree; // This variable stores the expected degree for this polinomial
    std::vector<ZZ> coefs;
    std::vector<std::vector<long> > polyCRT; // Must be initialized by crt()
    long *d_polyCRT; // Must be initialized on CRTSPACING definition and updated by crt(), if needed
    ZZ mod;
    Polynomial *phi;

    bool ON_COPY;
    bool HOST_IS_UPDATED;
    bool DEVICE_IS_UPDATE;
    //Functions and methods

};
#endif
