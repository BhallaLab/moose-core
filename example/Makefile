#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = example.o

OBJ = \
	Average.o	\

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
Average.o:	Average.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(GSL_FLAGS) -I../basecode $< -c

$(TARGET):		$(OBJ) $(GSL_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(GSL_OBJ)  $(GSL_LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
