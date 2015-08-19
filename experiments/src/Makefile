all: bits32 bits64

bits32:
	nvcc -O2 -g -G add32.cu -o ../bin/add32 -lntl -lgmp -Xcompiler -O2 -m64
bits64:
	nvcc -O2 -g -G add64.cu -o ../bin/add64 -lntl -lgmp -Xcompiler -O2 -m64
addmul64:
	nvcc -O2 -g -G addmul64.cu -o ../bin/addmul64 -lntl -lgmp -Xcompiler -O2 -m64

addmul32:
	nvcc -O2 -g -G addmul32.cu -o ../bin/addmul32 -lntl -lgmp -Xcompiler -O2 -m64

