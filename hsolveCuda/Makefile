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

SUBDIR = \
	cudaLibrary \
	testMoose	\
	testHines	\

TARGET = _hsolveCuda.o

OBJ = \
	HSolveCuda.o 	

HEADERS = \
	../basecode/header.h

default: $(TARGET) 

$(OBJ)	: $(HEADERS)
testHSolveCuda.o: testHSolveCuda.cpp
HSolveCuda.o: HSolveCuda.cpp HSolveCuda.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I/usr/local/cuda/include -I. -I../basecode -I../msg $< -c 


$(TARGET): subdirs $(OBJ) $(SMOLDYN_OBJ) $(HEADERS) 
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) $(GSL_LIBS)

subdirs:
	@(for i in $(SUBDIR) $(PARALLEL_DIR); do $(MAKE) -C $$i; done)

clean:
	@(for i in $(SUBDIR) ; do $(MAKE) -C $$i clean;  done)
	rm -f *.o $(TARGET)
