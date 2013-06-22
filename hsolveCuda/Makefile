#/**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Multiscale Object Oriented Simulation Environment.
# **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
# ** It is made available under the terms of the
# ** GNU Lesser General Public License version 2.1
# ** See the file COPYING.LIB for the full notice.
# **********************************************************************/
 
 SUBDIR = \
	cudaLibrary \
	test	\

TARGET = _hsolveCuda.o

OBJ = \
	HSolveCuda.o 	\
	testHSolveCuda.o \

HEADERS = \
	../basecode/header.h

default: $(TARGET) 

$(OBJ)	: $(HEADERS)
testHSolveCuda.o: testHSolveCuda.cpp
HSolveCuda.o: HSolveCuda.cpp HSolveCuda.h
test.o: test.cpp

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I/usr/local/cuda/include -I. -I../basecode -I../msg $< -c 


$(TARGET): subdirs $(OBJ) $(SMOLDYN_OBJ) $(HEADERS) 
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) $(GSL_LIBS)

subdirs:
	@(for i in $(SUBDIR) $(PARALLEL_DIR); do $(MAKE) -C $$i; done)

clean:
	@(for i in $(SUBDIR) ; do $(MAKE) -C $$i clean;  done)
	rm -f *.o $(TARGET)
