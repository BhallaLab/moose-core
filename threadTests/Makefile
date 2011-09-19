#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = barriers

OBJ = \
	FuncBarrier.o \
	main.o	\

HEADERS = \
	FuncBarrier.h \

#CXXFLAGS = -g -pthread -Wno-long-long -pedantic
CXXFLAGS = -O3 -pthread -Wno-long-long -pedantic

LIBS =  -lm -lpthread -L/usr/lib -L/usr/local/lib

default: $(TARGET)

$(OBJ)	: $(HEADERS)

.cpp.o:
	$(CXX) $(CXXFLAGS) -I. $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(CXX) -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
