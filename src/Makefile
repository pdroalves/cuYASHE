OPT = -O0

CUDA_CC = nvcc $(OPT) -G -g -std=c++11
CC = g++ -std=c++11 -g -Wall -Wfatal-errors -m64 $(OPT)

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

tests:  polynomial.o cuda_functions.o test.o test_ctpt.o test_ct_mul.o test_ntt.o test_mul.o test_crt.o test_reduce.o ciphertext.o distribution.o yashe.o common.o integer.o
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN) test.o polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o common.o integer.o -lcufft -Xcompiler $(OPENMP) $(NTL) -lboost_unit_test_framework
	# $(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN)_ctpt test_ctpt.o polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o common.o integer.o -lcufft -Xcompiler $(OPENMP) $(NTL)
	# $(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN)_crt test_crt.o polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o common.o integer.o -lcufft -Xcompiler $(OPENMP) $(NTL)
	# $(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN)_reduce test_reduce.o polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o common.o integer.o -lcufft -Xcompiler $(OPENMP) $(NTL)
	# $(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN)_ct_mul test_ct_mul.o polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o common.o integer.o -lcufft -Xcompiler $(OPENMP) $(NTL)
	# $(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN)_mul test_mul.o polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o common.o integer.o -lcufft -Xcompiler $(OPENMP) $(NTL)
	# $(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN)_ntt test_ntt.o  polynomial.o cuda_functions.o ciphertext.o distribution.o yashe.o common.o integer.o -lcufft -Xcompiler $(OPENMP) $(NTL)
	
test_fft:
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(TESTBIN)_fft test_fft.cu -lcufft

benchmarks:benchmark_yashe.o yashe.o benchmark_polynomial.o polynomial.o cuda_functions.o distribution.o ciphertext.o integer.o
		$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA)  -o $(BENCHMARKPOLYBIN) benchmark_polynomial.o polynomial.o cuda_functions.o common.o integer.o -lcufft -Xcompiler $(OPENMP) $(NTL) $(LTIME_MEASURE)
		$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA)  -o $(BENCHMARKYASHEBIN) benchmark_yashe.o yashe.o  distribution.o polynomial.o ciphertext.o integer.o cuda_functions.o common.o -lcufft -Xcompiler $(OPENMP) $(NTL) $(LTIME_MEASURE)

benchmark_polynomial.o:benchmark_polynomial.cpp
		$(CC) -c benchmark_polynomial.cpp -o benchmark_polynomial.o $(OPENMP) $(LCUDA) $(ICUDA) $(LTIME_MEASURE)

benchmark_yashe.o:benchmark_yashe.cpp
		$(CC) -c benchmark_yashe.cpp -o benchmark_yashe.o $(OPENMP) $(LCUDA) $(ICUDA) $(LTIME_MEASURE)

polynomial.o:polynomial.cpp
	$(CC) -c polynomial.cpp -o polynomial.o $(OPENMP) $(LCUDA) $(ICUDA) $(NTL)

ciphertext.o:ciphertext.cpp
	$(CC) -c ciphertext.cpp -o ciphertext.o $(NTL) $(LCUDA) $(OPENMP) $(ICUDA)

integer.o:integer.cpp
	$(CC) -c integer.cpp -o integer.o $(NTL) $(LCUDA) $(OPENMP) $(ICUDA)

cuda_functions.o:cuda_functions.cu
	$(CUDA_CC) $(CUDA_ARCH) -c cuda_functions.cu $(LCUDA) $(ICUDA) -lcufft -Xcompiler $(OPENMP)

cuda_bn.o:cuda_bn.cu
	$(CUDA_CC) $(CUDA_ARCH) -c cuda_bn.cu $(LCUDA) $(ICUDA) -Xcompiler $(NTL)


distribution.o:distribution.cpp
	$(CC) -c distribution.cpp -o distribution.o $(NTL) $(OPENMP)  $(LCUDA) $(ICUDA)

yashe.o:yashe.cpp
	$(CC) -c yashe.cpp -o yashe.o $(NTL) $(OPENMP) $(LCUDA) $(ICUDA)

common.o:common.cpp
	$(CC) -c common.cpp -o common.o $(NTL) $(OPENMP) $(LCUDA) $(ICUDA)

test_ctpt.o: test_ctpt.cu
	# $(CUDA_CC) -c test_ctpt.cu -o test_ctpt.o  $(NTL)

test_crt.o: test_crt.cu
	# $(CUDA_CC) -c test_crt.cu -o test_crt.o  $(NTL)

test_mul.o:
	# $(CUDA_CC) -c test_mul.cu -o test_mul.o  $(NTL)

test_reduce.o:
	# $(CUDA_CC) -c test_reduce.cu -o test_reduce.o  $(NTL)

test_ct_mul.o:
	# $(CUDA_CC) -c test_ct_mul.cu -o test_ct_mul.o  $(NTL)


test_ntt.o:test_ntt.cu
	# $(CUDA_CC) -c test_ntt.cu -o test_ntt.o  $(NTL)


test.o:test.cu
	$(CUDA_CC) -c test.cu -o test.o  -lboost_unit_test_framework $(NTL)

clean:
	rm -f $(BIN) $(TESTBIN) $(BENCHMARKPOLYBIN) $(BENCHMARKYASHEBIN) integer.o test_crt.o  test_ct_mul.o  test_ctpt.o  test_mul.o common.o benchmark_polynomial.o benchmark_yashe.o polynomial.o tests.o test_reduce.o ciphertext.o cuda_functions.o kernels.o distribution.o yashe.o test.o
