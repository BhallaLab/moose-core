#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2003- 2006 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

CXXFLAGS = -g -pthread -Wall -Wno-long-long -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER
USE_MPI = 1
LIBS =	-lm -lpthread

# For 64 bit Linux systems add paths to 64 bit libraries 
ifeq ($(OSTYPE),linux)
CXXFLAGS += -DLINUX
ifeq ($(MACHINE),x86_64)
LIBS+= -L/lib64 -L/usr/lib64
endif
endif

# Here we automagically change compilers to deal with MPI.
ifdef USE_MPI
	CXX = mpicxx
else
	CXX = g++
endif

TARGET = lt

OBJ = \
	main.o	\
	Tracker.o \
	EventLoop.o \

HEADERS = \
	Tracker.h \
	ProcInfo.h

default: $(TARGET)

$(OBJ)	: $(HEADERS)

.cpp.o:
	$(CXX) $(CXXFLAGS) $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(CXX)  $(CXXFLAGS) $(OBJ) -o $@ $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
