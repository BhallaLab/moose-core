#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = parallel.o

OBJ = \
	PostMaster.o	\
	ProxyElement.o	\
	AsyncDestFinfo.o	\
	TestPostMaster.o	\
	TestParOps.o	\

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
PostMaster.o:	PostMaster.h
TestPostMaster.o:	PostMaster.h

# Now CXX value is decided in top-level Makefile
#ifneq (,$(findstring DUSE_MPI,$(CXXFLAGS)))
#    CXX = mpicxx
#else
#    CXX = g++
#endif

# This Wno-long-long flag is needed to compile mpich_shmem on Ubuntu 8.04
# It is a problem with all recent versions of mpich, which define MPI_Offset
# as a long long, which is illegal in ISO C++.
.cpp.o:
	$(CXX) $(CXXFLAGS) -Wno-long-long -I.. -I../basecode $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
