OPT = -O0

CUDA_CC = nvcc $(OPT) -G -g -std=c++11
CC = g++ -std=c++11 -g -Wall -Wfatal-errors -m64 $(OPT)

OPENMP = -fopenmp
# OPENMP = 

LTIME_MEASURE = -lrt
NTL = -Intl -lntl -lgmp

LCUDA = -lcuda -lcudart -lcudadevrt -L/usr/local/cuda/lib64 -lcurand
ICUDA = -I/usr/local/cuda/include
CUDA_ARCH = -arch=sm_35
ICUYASHE = -I$(HOME)/git/cuyashe/src
LCUYASHE = -L$(HOME)/git/cuyashe/src
CUYASHE = $(HOME)/git/cuyashe/src

BIN = bin
SRC = src
INSTALL_BENCHMARK = $(BIN)/benchmark
INSTALL_CMP = $(BIN)/result_cmp
INSTALL_CUFFT = $(BIN)/cufft
INSTALL_CUFFTMUL = $(BIN)/cufft_with_mul
INSTALL_CUFFTMUL_SINGLE = $(BIN)/cufft_with_mul_single

all:benchmark.o result_cmp.o cufft.o cufft_with_mul.o cufft_with_mul_single.o
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) $(ICUYASHE) -o $(INSTALL_BENCHMARK) benchmark.o $(CUYASHE)/cuda_functions.o $(CUYASHE)/cuda_bn.o $(CUYASHE)/distribution.o $(CUYASHE)/polynomial.o $(CUYASHE)/common.o $(CUYASHE)/cuda_distribution.o $(CUYASHE)/integer.o -lcufft -lcurand -Xcompiler $(OPENMP) $(NTL) $(LTIME_MEASURE)
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) $(ICUYASHE) -o $(INSTALL_CMP) result_cmp.o $(CUYASHE)/cuda_functions.o $(CUYASHE)/cuda_bn.o $(CUYASHE)/distribution.o $(CUYASHE)/polynomial.o $(CUYASHE)/common.o $(CUYASHE)/cuda_distribution.o $(CUYASHE)/integer.o -lcufft -lcurand -Xcompiler $(OPENMP) $(NTL) $(LTIME_MEASURE)
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(INSTALL_CUFFT) cufft.o -lcufft -Xcompiler $(OPENMP) $(LTIME_MEASURE)
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(INSTALL_CUFFTMUL) cufft_with_mul.o -lcufft -Xcompiler $(OPENMP) $(LTIME_MEASURE)
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) -o $(INSTALL_CUFFTMUL_SINGLE) cufft_with_mul_single.o -lcufft -Xcompiler $(OPENMP) $(LTIME_MEASURE)

benchmark.o:$(SRC)/benchmark.cpp
	$(CC) -c  $(SRC)/benchmark.cpp -o benchmark.o $(OPENMP) $(LCUDA) $(ICUDA) $(LTIME_MEASURE) $(ICUYASHE) $(NTL)

result_cmp.o:$(SRC)/result_cmp.cpp
	$(CC) -c  $(SRC)/result_cmp.cpp -o result_cmp.o $(OPENMP) $(LCUDA) $(ICUDA) $(LTIME_MEASURE) $(ICUYASHE) $(NTL)

cufft.o:$(SRC)/cufft.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cufft.cu -Xcompiler $(NTL) -Xcompiler $(OPENMP) 
	
cufft_with_mul.o:$(SRC)/cufft_with_mul.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cufft_with_mul.cu -Xcompiler $(NTL) -Xcompiler $(OPENMP) 

cufft_with_mul_single.o:$(SRC)/cufft_with_mul_single.cu
	$(CUDA_CC) $(CUDA_ARCH) -c $(SRC)/cufft_with_mul_single.cu -Xcompiler $(NTL) -Xcompiler $(OPENMP) 

# polynomial.o:polynomial.cpp
# 	$(CC) -c $SRC/polynomial.cpp -o polynomial.o $(OPENMP) $(LCUDA) $(ICUDA) $(NTL) $(ICUYASHE)

# cuda_functions.o:cuda_functions.cu
# 	$(CUDA_CC) $(CUDA_ARCH) -c $SRC/cuda_functions.cu $(LCUDA) $(ICUDA) -lcufft --relocatable-device-code true $(ICUYASHE) -Xcompiler $(OPENMP)

# distribution.o:distribution.cpp
# 	$(CC) -c distribution.cpp -o distribution.o $(NTL) $(OPENMP) -lcurand  $(LCUDA) $(ICUDA) $(ICUYASHE)

# common.o:common.cpp
# 	$(CC) -c common.cpp -o common.o $(NTL) $(OPENMP) $(LCUDA) $(ICUDA) $(ICUYASHE)
