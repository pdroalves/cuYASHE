#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <assert.h>
#include "polynomial.h"
#include <cuda.h>
#include <curand.h>

enum kind_t
{
  DISCRETE_GAUSSIAN,
  BINARY,
  NARROW,
  UNIFORMLY,
  KINDS_COUNT
};

class Distribution{
  private:
  int kind;
  float gaussian_std_deviation;
  int gaussian_bound;
  curandGenerator_t gen;

  public:
  Distribution(kind_t kind, float std_dev, int bound){
    assert(kind == DISCRETE_GAUSSIAN);
    this->kind = kind;
    this->gaussian_std_deviation = std_dev;
    this->gaussian_bound = bound;

    curandStatus_t result = curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT);
    assert(result == CURAND_STATUS_SUCCESS);

    result = curandSetPseudoRandomGeneratorSeed(gen, 
                1234ULL);
    assert(result == CURAND_STATUS_SUCCESS);
  }
  Distribution(kind_t kind){
    assert(kind != DISCRETE_GAUSSIAN);
    assert(kind < KINDS_COUNT);
    this->kind = kind;

    curandStatus_t result = curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT);

    result = curandSetPseudoRandomGeneratorSeed(gen, 
                1234ULL);
    assert(result == CURAND_STATUS_SUCCESS);
  }
  Distribution(){
    this->kind = UNIFORMLY;

    curandStatus_t result = curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT);
    assert(result == CURAND_STATUS_SUCCESS);

    result = curandSetPseudoRandomGeneratorSeed(gen, 
                1234ULL);
    assert(result == CURAND_STATUS_SUCCESS);
  }
  Polynomial get_sample(int degree);
private:
  void callCuGetUniformSample(cuyasheint_t *array, int N);
  void callCuGetNormalSample(cuyasheint_t *array, int N, float mean, float stddev);
};
#endif
