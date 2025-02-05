CUDA_HOME   = /Soft/cuda/11.2.1

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib

ALL = heat-CUDA
all: $(ALL)

kernels.o: kernels.cu
	$(NVCC) -c $(NVCCFLAGS) $+ $(LFLAGS) -o $@

heat-CUDA: heat-CUDA.cu kernels.o
	$(NVCC) $(NVCCFLAGS) $+ $(LFLAGS) -o $@

clean:
	rm -fr $(ALL) *.o *~ *.ppm *.ps *.txt sub*.e* sub*.o* *.prv *.pcf *.row TRACE.mpits set-0 *.sym