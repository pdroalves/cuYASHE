all: stockham stockham_ntt

addmul64:
	nvcc -O2 -g -G addmul64.cu -o ../bin/addmul64 -lntl -lgmp -Xcompiler -O2 -m64

addmul32:
	nvcc -O2 -g -G addmul32.cu -o ../bin/addmul32 -lntl -lgmp -Xcompiler -O2 -m64

ct_ntt:cooley_tukey_ntt.o cooley_tukey_kernels.o
	nvcc  -g -G -arch=sm_35 -rdc=true -o ../bin/cocooley_tukey_ntt cooley_tukey_ntt.o cooley_tukey_kernels.o -lntl -lgmp -O2 -m64 -lcuda -I/usr/local/cuda/include

cooley_tukey_ntt.o: cooley_tukey_ntt.cpp
	g++ -O2 -g -c cooley_tukey_ntt.cpp -o cooley_tukey_ntt.o -lntl -lgmp -m64 -lcuda -I/usr/local/cuda/include

cooley_tukey_kernels.o:cooley_tukey_kernels.cu
	nvcc -O2 -g -G -arch=sm_35 -rdc=true -c cooley_tukey_kernels.cu -o cooley_tukey_kernels.o  -lntl -lgmp -Xcompiler -O2 -m64 -lcudadevrt

stockham:
	nvcc -O0 -arch=sm_35 -G -g ./stockham.cpp ./stockham_reference.cu -o stockham_reference -lm -std=c++11
	# nvcc -O0 -arch=sm_35 -G -g ./stockham.cpp ./stockham_global.cu -o stockham_global -lm -std=c++11

stockham_ntt:
	nvcc -O0 -arch=sm_35 -G -g ./stockham_ntt.cpp ./stockham_ntt_reference.cu -o stockham_ntt_reference -lm -std=c++11 -lntl -lgmp
