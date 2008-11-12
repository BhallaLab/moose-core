#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = scheduling.o

OBJ = \
	ClockJob.o	\
	Tick.o	\
	ParTick.o	\
	SchedTests.o	\

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
ClockJob.o:	ClockJob.h ../basecode/Ftype2.h ../basecode/Ftype3.h
Tick.o:	Tick.h
ParTick.o:	Tick.h ParTick.h

.cpp.o:
	$(CXX) $(CXXFLAGS) -I../basecode -I.. $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) 
clean:
	-rm -f *.o $(TARGET) core core.*
