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
TARGET = hsolve.o

OBJ = \
	HSolveWrapper.o \
	HSolve.o \

HEADERS = \
	./ChCompartment.h \
	../basecode/header.h \
	../biophysics/Compartment.h \
	../basecode/Element.h \
	../basecode/Field.h \
	../basecode/Finfo.h \

default: $(TARGET)

$(OBJ) : $(HEADERS)
HSolveWrapper.o: HSolve.h HSolveWrapper.h
.cpp.o:
	$(CXX) $(CFLAGS) -I../basecode $< -c

$(TARGET):	$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*


