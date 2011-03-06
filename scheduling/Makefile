#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2009 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = scheduling.o

OBJ = \
	Clock.o	\
	Tick.o	\
	TickMgr.o	\
	testScheduling.o \

HEADERS = \
	../basecode/header.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
Clock.o:	Tick.h TickPtr.h TickMgr.h Clock.h
Tick.o:	Tick.h TickPtr.h TickMgr.h Clock.h
TickMgr.o:	Tick.h TickMgr.h
testScheduling.o:	Tick.h TickPtr.h Clock.h ../basecode/SetGet.h


.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
