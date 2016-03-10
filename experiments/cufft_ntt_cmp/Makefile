OPT = -O3

CUDA_CC = nvcc $(OPT) -G -g -std=c++11
CC = g++ -std=c++11 -g -Wall -Wfatal-errors -m64 $(OPT)

OPENMP = -fopenmp
# OPENMP = 

LTIME_MEASURE = -lrt
NTL = -Intl -lntl -lgmp

LCUDA = -lcuda -lcudart -lcudadevrt -L/usr/local/cuda/lib64 -lcurand
ICUDA = -I/usr/local/cuda/include
CUDA_ARCH = -arch=sm_35
ICUYASHE = -I/home/pdroalves/mestrado/cuYASHE/src
LCUYASHE = -L/home/pdroalves/mestrado/cuYASHE/src
CUYASHE = /home/pdroalves/mestrado/cuYASHE/src

BIN = bin
SRC = src
INSTALL_BENCHMARK = $(BIN)/benchmark

all:benchmark.o 
	$(CUDA_CC) $(CUDA_ARCH) $(LCUDA) $(ICUDA) $(ICUYASHE) -o $(INSTALL_BENCHMARK) benchmark.o $(CUYASHE)/cuda_functions.o $(CUYASHE)/cuda_bn.o $(CUYASHE)/distribution.o $(CUYASHE)/polynomial.o $(CUYASHE)/common.o $(CUYASHE)/cuda_distribution.o $(CUYASHE)/integer.o -lcufft -lcurand -Xcompiler $(OPENMP) $(NTL) $(LTIME_MEASURE)

benchmark.o:$(SRC)/benchmark.cpp
	$(CC) -c  $(SRC)/benchmark.cpp -o benchmark.o $(OPENMP) $(LCUDA) $(ICUDA) $(LTIME_MEASURE) $(ICUYASHE) $(NTL)

# polynomial.o:polynomial.cpp
# 	$(CC) -c $SRC/polynomial.cpp -o polynomial.o $(OPENMP) $(LCUDA) $(ICUDA) $(NTL) $(ICUYASHE)

# cuda_functions.o:cuda_functions.cu
# 	$(CUDA_CC) $(CUDA_ARCH) -c $SRC/cuda_functions.cu $(LCUDA) $(ICUDA) -lcufft --relocatable-device-code true $(ICUYASHE) -Xcompiler $(OPENMP)

# distribution.o:distribution.cpp
# 	$(CC) -c distribution.cpp -o distribution.o $(NTL) $(OPENMP) -lcurand  $(LCUDA) $(ICUDA) $(ICUYASHE)

# common.o:common.cpp
# 	$(CC) -c common.cpp -o common.o $(NTL) $(OPENMP) $(LCUDA) $(ICUDA) $(ICUYASHE)
