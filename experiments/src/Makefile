BIN=../bin

all: stockham stockham_ntt

addmul64:
	nvcc -O2 -g -G addmul64.cu -o ../bin/addmul64 -lntl -lgmp -Xcompiler -O2 -m64

addmul32:
	nvcc -O2 -g -G addmul32.cu -o ../bin/addmul32 -lntl -lgmp -Xcompiler -O2 -m64

stockham:
	g++ -O0 -g ./stockham.cpp ./stockham_reference.cpp -o $(BIN)/stockham -lm -std=c++11

stockham_ntt:
	g++ -O0 -g ./stockham_ntt.cpp ./stockham_ntt_reference.cpp -o $(BIN)/stockham_ntt -lm -std=c++11 -lntl -lgmp
