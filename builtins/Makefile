#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = builtins.o

OBJ = \
	Interpol.o	\
	Interpol2D.o	\
	Table.o	\
	TimeTable.o \
	AscFile.o \
	Calculator.o

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Interpol.o:	Interpol.h ../basecode/Ftype2.h
Interpol2D.o:	Interpol.h Interpol2D.h ../basecode/Ftype2.h
Table.o:	Table.h Interpol.h ../basecode/Ftype2.h ../basecode/Ftype3.h
TimeTable.o:    TimeTable.h
AscFile.o:      AscFile.h
Calculator.o:	Calculator.h

.cpp.o:
	$(CXX) $(CXXFLAGS) -I.. -I../basecode $< -c

#$(TARGET):		$(OBJ) $(HEADERS)
#	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
