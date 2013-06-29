#/**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Multiscale Object Oriented Simulation Environment.
# **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
# ** It is made available under the terms of the
# ** GNU Lesser General Public License version 2.1
# ** See the file COPYING.LIB for the full notice.
# **********************************************************************/

#TODO: imported from main Makefile for test, after we can remove it
BUILD=debug
ifeq ($(BUILD),debug)
CXXFLAGS = -g -pthread -fpermissive -fno-strict-aliasing -fPIC -fno-inline-functions -Wall -Wno-long-long -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER
USE_GSL = 1
endif
####################################################################


CU_SRCS += 	CudaModule.cu 			
		
CU_DEPS += CudaModule.d 

COBJS += CudaModule.o 		\
		 HinesGpu.o 		\
		 GpuSimulationControl.o

OBJS += SynapticChannels.o  \
		ActiveChannels.o  	\
		HinesMatrix.o  		\
		PlatformFunctions.o \
		Connections.o  		\
		SpikeStatistics.o  	\
		NeuronInfoWriter.o  \
		PerformSimulation.o \
		
libmooseCudaLibrary.so: $(OBJS) $(COBJS)
	nvcc -shared -link -o  "libmooseCudaLibrary.so" $(OBJS) $(COBJS)

.PHONY: all
all: libmooseCudaLibrary.so
default: all

SynapticChannels.o: 	SynapticChannels.cpp 	SynapticChannels.hpp
ActiveChannels.o:		ActiveChannels.cpp 		ActiveChannels.hpp
HinesMatrix.o:			HinesMatrix.cpp			HinesMatrix.hpp
PlatformFunctions.o:	PlatformFunctions.cpp	PlatformFunctions.hpp
Connections.o:			Connections.cpp			Connections.hpp
SpikeStatistics.o:  	SpikeStatistics.cpp		SpikeStatistics.hpp
NeuronInfoWriter.o: 	NeuronInfoWriter.cpp	NeuronInfoWriter.hpp
PerformSimulation.o:	PerformSimulation.cpp	PerformSimulation.hpp

CudaModule.o:	CudaModule.cu CudaModule.h
HinesGpu.o:		HinesGpu.cu
GpuSimulationControl.o: GpuSimulationControl.cu GpuSimulationControl.hpp

$(OBJS):
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I/usr/local/cuda/include -I. -I../basecode -I../msg $< -c 
	

$(COBJS):
	nvcc --compile -G -O0 -Xcompiler -fPIC -g -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<" #-lpthread --compiler-options -O3 --ptxas-options=-v -m64 

.PHONY: clean
clean:
	rm -rf $(CU_DEPS) $(OBJS) $(COBJS) libmooseCudaLibrary.so
