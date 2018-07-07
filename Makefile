CC=nvcc

all: sar.bin

sar.bin: sar.o libbmp.o
	nvcc sar.o libbmp.o -o sar.bin

sar.o: 
	nvcc -c sar.cu -o sar.o

libbmp.o: 
	nvcc -c libbmp.c -o libbmp.o

clean:
	rm -rf a.out sar.bin *.o

run:
	make clean
	make
	nvprof ./sar.bin 1999.08.14.bmp 1996.08.24.bmp > test.csv