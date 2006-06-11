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

TARGET = mpp

TEST_TARGET = mpptest

CXX = g++ -g

OBJ = \
	Mpp.o \
	Field.o \
	Src.o \
	Dest.o \
	Synapse.o \
	Conn.o \
	Shared.o \

TESTOBJ = \
	Mpp.o \
	Field.o \
	MppTest.o \
	FieldTest.o \

HEADERS = \
	Mpp.h \
	Field.h \
	Src.h \
	Dest.h \
	Synapse.h \
	Conn.h \
	Shared.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS) 

$(TESTOBJ)	: $(HEADERS)

.cpp.o:
	$(CXX) $(CFLAGS) $< -c

$(TARGET):		$(OBJ) $(HEADERS)
		$(CXX) $(CFLAGS) $(OBJ) $(LIBS) -o $(TARGET)

$(TEST_TARGET):		$(TESTOBJ) $(HEADERS)
		$(CXX) $(CFLAGS) $(OBJ) $(LIBS) -o $(TARGET)

clean:
	-rm -f *.o $(TARGET) $(TEST_TARGET) core core.*
