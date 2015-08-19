CUDA_CC = nvcc -O0 -G -g
CC = g++ -std=c++11 -g -Wall -m64

OPENMP = -fopenmp

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

# all:main all-tests benchmark
all: tests benchmarkpoly

tests: polynomial.o cuda_functions.o test.o
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA)  -o $(TESTBIN) test.o polynomial.o cuda_functions.o -Xcompiler $(OPENMP) $(NTL) -lboost_unit_test_framework

# main:main.o polynomial.o ciphertext.o cuda_functions.o kernels.o distribution.o yashe.o
# 	$(CUDA_CC) $(LCUDA) $(ICUDA)  -o $(BIN) main.o polynomial.o ciphertext.o cuda_functions.o kernels.o distribution.o yashe.o -lcufft -Xcompiler $(OPENMP) $(NTL)

benchmarkpoly:benchmark_polynomial.o polynomial.o cuda_functions.o
		$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA)  -o $(BENCHMARKPOLYBIN) benchmark_polynomial.o polynomial.o cuda_functions.o  -Xcompiler $(OPENMP) $(NTL) $(LTIME_MEASURE)

# main.o:main.cpp
# 	$(CC) -c main.cpp -o main.o $(LCUDA) $(ICUDA)

benchmark_polynomial.o:benchmark_polynomial.cpp
		$(CC) -c benchmark_polynomial.cpp -o benchmark_polynomial.o $(OPENMP) $(LCUDA) $(ICUDA) $(LTIME_MEASURE)

# benchmark_yashe.o:benchmark_yashe.cpp
# 		$(CC) -c benchmark_yashe.cpp -o benchmark_yashe.o $(LCUDA) $(ICUDA) $(LTIME_MEASURE)

polynomial.o:polynomial.cpp
	$(CC) -c polynomial.cpp -o polynomial.o $(OPENMP) $(LCUDA) $(ICUDA) $(NTL)

# ciphertext.o:ciphertext.cpp
# 	$(CC) -c ciphertext.cpp -o ciphertext.o $(NTL) $(LCUDA) $(ICUDA)

cuda_functions.o:cuda_functions.cu
	$(CUDA_CC) $(CUDA_ARCH) -c cuda_functions.cu $(LCUDA) $(ICUDA) -Xcompiler -m64

# kernels.o:kernels.cu
# 	$(CUDA_CC) $(CUDA_ARCH) -c kernels.cu -lcufft
#
# distribution.o:distribution.cpp
# 	$(CC) -c distribution.cpp -o distribution.o $(NTL) $(LCUDA) $(ICUDA)
#
# yashe.o:yashe.cpp
# 	$(CC) -c yashe.cpp -o yashe.o $(NTL) $(LCUDA) $(ICUDA)

test_ntt: cuda_functions.o
	$(CUDA_CC) $(CUDA_ARCH) test_ntt.cu cuda_functions.cu -o ../bin/test_ntt -lntl -lgmp -Xcompiler -m64


test_mul: polynomial.o cuda_functions.o
	$(CUDA_CC) $(CUDA_ARCH) -G -g test_mul.cu polynomial.cpp cuda_functions.cu -o ../bin/test_mul -lntl -lgmp
test.o:test.cu
	$(CUDA_CC) -c test.cu -o test.o  -lboost_unit_test_framework $(NTL)

clean:
	rm -f $(BIN) $(TESTBIN) $(BENCHMARKPOLYBIN) $(BENCHMARKYASHEBIN) main.o benchmark_polynomial.o benchmark_yashe.o polynomial.o tests.o ciphertext.o cuda_functions.o kernels.o distribution.o yashe.o test.o
