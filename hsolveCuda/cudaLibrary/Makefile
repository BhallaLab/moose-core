#/**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Multiscale Object Oriented Simulation Environment.
# **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
# ** It is made available under the terms of the
# ** GNU Lesser General Public License version 2.1
# ** See the file COPYING.LIB for the full notice.
# **********************************************************************/
CU_SRCS += CudaModule.cu 

CU_DEPS += CudaModule.d 

OBJS += CudaModule.o 

%.o: %.cu
	nvcc -G -g -O0 -Xcompiler -fPIC -gencode arch=compute_30,code=sm_30 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -Xcompiler -fPIC -g -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	
all: libmooseCudaLibrary.so


libmooseCudaLibrary.so: $(OBJS) 
	nvcc -shared -link -o  "libmooseCudaLibrary.so" $(OBJS) 

clean:
	rm -rf $(CU_DEPS)$(OBJS) libmooseCudaLibrary.so
