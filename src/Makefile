OBJS = ideal.o

NVCC = /usr/local/cuda/bin/nvcc

CPICFLAGS = -fpic
RPATH = 

#compiler/preprocessor options
EXTRA_INCLUDES =  -I/usr/lib64/R/include  -I./Random123-1.09/include

EXTRA_LIBS =  -lcuda 

CFLAGS = -g -O2  
LDFLAGS = 

EXTRA_CFLAGS =  
EXTRA_CPPFLAGS = -I/usr/local/include

EXTRA_LDFLAGS =    

TARGETS = idealcu.so

all: $(TARGETS) 

$(TARGETS): $(OBJS)
	$(NVCC) -shared $(RPATH) $(LDFLAGS) $(EXTRA_LDFLAGS) $(LIBS) $(EXTRA_LIBS) $(OBJS) -o $@

$(OBJS): %.o: %.cu
	$(NVCC) -c -Xcompiler "$(CPICFLAGS) $(CPPFLAGS) $(EXTRA_CPPFLAGS) $(CFLAGS) $(EXTRA_CFLAGS)" -D_FORCE_INLINES -I. $(INCLUDES) $(EXTRA_INCLUDES) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean


