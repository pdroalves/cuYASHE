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

tests:  polynomial.o cuda_functions.o test.o test_ctpt.o test_crt.o ciphertext.o distribution.o yashe.o 
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN) test.o polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o -lcufft -Xcompiler $(OPENMP) $(NTL) -lboost_unit_test_framework
	# $(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN)_ctpt test_ctpt.o polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o -lcufft -Xcompiler $(OPENMP) $(NTL)
	# $(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN)_crt test_crt.o polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o -lcufft -Xcompiler $(OPENMP) $(NTL)

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

common.o:common.cpp
	$(CC) -c common.cpp -o common.o $(NTL) $(OPENMP) $(LCUDA) $(ICUDA)

test_ctpt.o: test_ctpt.cu
	$(CUDA_CC) -c test_ctpt.cu -o test_ctpt.o  $(NTL)

test_crt.o: test_crt.cu
	$(CUDA_CC) -c test_crt.cu -o test_crt.o  $(NTL)

test_mul: polynomial.o cuda_functions.o
	$(CUDA_CC) $(CUDA_ARCH) -G -g test_mul.cu polynomial.cpp cuda_functions.cu -o ../bin/test_mul -lntl -lgmp

test.o:test.cu
	$(CUDA_CC) -c test.cu -o test.o  -lboost_unit_test_framework $(NTL)

clean:
	rm -f $(BIN) $(TESTBIN) $(BENCHMARKPOLYBIN) $(BENCHMARKYASHEBIN) main.o benchmark_polynomial.o benchmark_yashe.o polynomial.o tests.o ciphertext.o cuda_functions.o kernels.o distribution.o yashe.o test.o
