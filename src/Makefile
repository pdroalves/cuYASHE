OPT = -O3

CUDA_CC = nvcc $(OPT) -G -g -std=c++11
CC = g++ -std=c++11 -g -Wall -Wfatal-errors -m64 $(OPT)

# OpenMP isn't useful here
OPENMP = -fopenmp
# OPENMP = 

NTL = -Intl -lntl -lgmp

LTIME_MEASURE = -lrt

#LCUDA = -L/usr/local/cuda/lib64
LCUDA = -lcuda
ICUDA = -I/usr/local/cuda/include
CUDA_ARCH = -arch=sm_35

INSTALL = ../bin
BIN = $(INSTALL)/main
TESTBIN = $(INSTALL)/test
BENCHMARKPOLYBIN= $(INSTALL)/benchmark_poly
BENCHMARKYASHEBIN= $(INSTALL)/benchmark_yashe

all: tests benchmarks

tests: polynomial.o cuda_functions.o test.o ciphertext.o distribution.o yashe.o
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA)  -o $(TESTBIN) test.o polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o -lcufft -Xcompiler $(OPENMP) $(NTL) -lboost_unit_test_framework

benchmarks:benchmark_yashe.o yashe.o benchmark_polynomial.o polynomial.o cuda_functions.o distribution.o ciphertext.o
		$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA)  -o $(BENCHMARKPOLYBIN) benchmark_polynomial.o polynomial.o cuda_functions.o -lcufft -Xcompiler $(OPENMP) $(NTL) $(LTIME_MEASURE)
		$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA)  -o $(BENCHMARKYASHEBIN) benchmark_yashe.o yashe.o  distribution.o polynomial.o ciphertext.o cuda_functions.o -lcufft -Xcompiler $(OPENMP) $(NTL) $(LTIME_MEASURE)

benchmark_polynomial.o:benchmark_polynomial.cpp
		$(CC) -c benchmark_polynomial.cpp -o benchmark_polynomial.o $(OPENMP) $(LCUDA) $(ICUDA) $(LTIME_MEASURE)

benchmark_yashe.o:benchmark_yashe.cpp
		$(CC) -c benchmark_yashe.cpp -o benchmark_yashe.o $(OPENMP) $(LCUDA) $(ICUDA) $(LTIME_MEASURE)

polynomial.o:polynomial.cpp
	$(CC) -c polynomial.cpp -o polynomial.o $(OPENMP) $(LCUDA) $(ICUDA) $(NTL)

ciphertext.o:ciphertext.cpp
	$(CC) -c ciphertext.cpp -o ciphertext.o $(NTL) $(LCUDA) $(OPENMP) $(ICUDA)

cuda_functions.o:cuda_functions.cu
	$(CUDA_CC) $(CUDA_ARCH) -c cuda_functions.cu $(LCUDA) $(ICUDA) -lcufft -Xcompiler -m64


distribution.o:distribution.cpp
	$(CC) -c distribution.cpp -o distribution.o $(NTL) $(OPENMP)  $(LCUDA) $(ICUDA)

yashe.o:yashe.cpp
	$(CC) -c yashe.cpp -o yashe.o $(NTL) $(OPENMP) $(LCUDA) $(ICUDA)

test_ntt: test_ntt.cu polynomial.o ciphertext.o yashe.o
	nvcc -O3 -G -g -std=c++11 -arch=sm_35 test_ntt.cu polynomial.cpp ciphertext.cpp yashe.o cuda_functions.cu distribution.cpp -o ../bin/test_ntt -lcufft -Xcompiler  -fopenmp -Intl -lntl -lgmp -m64


test_mul: polynomial.o cuda_functions.o
	$(CUDA_CC) $(CUDA_ARCH) -G -g test_mul.cu polynomial.cpp cuda_functions.cu -o ../bin/test_mul -lntl -lgmp
test.o:test.cu
	$(CUDA_CC) -c test.cu -o test.o  -lboost_unit_test_framework $(NTL)

clean:
	rm -f $(BIN) $(TESTBIN) $(BENCHMARKPOLYBIN) $(BENCHMARKYASHEBIN) main.o benchmark_polynomial.o benchmark_yashe.o polynomial.o tests.o ciphertext.o cuda_functions.o kernels.o distribution.o yashe.o test.o
