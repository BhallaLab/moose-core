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

TARGET = basecode.o
CXXFLAGS += -I.. -I../kinetics -I../biophysics
OBJ = \
	Element.o	\
	Eref.o	\
	Finfo.o	\
	Msg.o	\
	SparseMatrix.o	\
	testSync.o	\
	main.o	\

HEADERS = \
	header.h \
	Element.h \
	Eref.h \
	Msg.h \
	SparseMatrix.h \
	Finfo.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
#TestBasecode.o:	Shell.h ShellWrapper.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
