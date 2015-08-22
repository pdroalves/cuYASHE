#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <assert.h>
#include "polynomial.h"

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

  public:
  Distribution(kind_t kind, float std_dev, int bound){
    assert(kind == DISCRETE_GAUSSIAN);
    this->kind = kind;
    this->gaussian_std_deviation = std_dev;
    this->gaussian_bound = bound;
  }
  Distribution(kind_t kind){
    assert(kind != DISCRETE_GAUSSIAN);
    assert(kind < KINDS_COUNT);
    this->kind = kind;
  }
  Distribution(){
    this->kind = UNIFORMLY;
  }
  Polynomial get_sample(int degree);
};
#endif
