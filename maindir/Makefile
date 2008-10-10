#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/
# $Id: $
#
# $Log: $
#

TARGET = maindir.o

OBJ = \
	main.o	\
	nonblock.o \
	init.o \
	mpiSetup.o \
	MuMPI.o

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \
	../basecode/Id.h 

default: $(TARGET)

$(OBJ)	: $(HEADERS)
MuMPI.o:	MuMPI.h

ifneq (,$(findstring DUSE_MPI,$(CFLAGS)))
    CXX = mpicxx -Wno-long-long
else
    CXX = g++
endif

.cpp.o:
	$(CXX) $(CFLAGS) -I.. -I../basecode $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
