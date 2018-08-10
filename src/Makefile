GCC = gcc
NVCC = /usr/local/cuda-6.5/bin/nvcc -ccbin $(GCC)
NVCCFLAGS = -m${OS_SIZE} ${ARCH_FLAGS} -Xcompiler -fPIC
NVCC_INC = -I/lib64/R/include  \
	   -I/usr/local/cuda-6.5/include -I./Random123-1.08/include 
SHLIB_FLAGS = -L/usr/local/cuda-6.5/targets/x86_64-linux/lib  -lcudart 

idealcu:
	$(NVCC) $(NVCFLAGS) $(NVCC_INC) --compiler-options -fPIC -c ideal.cu
	R CMD SHLIB $(SHLIB_FLAGS) ideal.o -o idealcu.so

clean:
	find . -name *~ -type f -delete	
	- rm *.o

clean_so:
	rm *.so

