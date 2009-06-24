#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = sbml_IO.o
#LIBS +=-lsbml  -L/usr/local/lib
OBJ = \
	SbmlReader.o	\
	SbmlWriter.o	\
	

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
SbmlReader.o:	SbmlReader.h ../element/Neutral.h
SbmlWriter.o:	SbmlWriter.h ../element/Neutral.h

.cpp.o:
	$(CXX) $(CXXFLAGS) -I../basecode -I../utility/randnum -I.. $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*

