#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = neuroML_IO.o
OBJ = \
	NeuromlReader.o	\
	Cable.o  	\
	NeuromlWriter.o	\

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
NeuromlReader.o:	NeuromlReader.h ../element/Neutral.h
NeuromlWriter.o:	NeuromlWriter.h ../element/Neutral.h
Cable.o:		Cable.h
//NBase.o:		NBase.h ../element/Neutral.h
//Segment.o:		Segment.h ../element/Neutral.h


.cpp.o:
	$(CXX) $(CXXFLAGS) -I../external/include -I/usr/include/libxml2 \
       	-I../basecode -I../utility/randnum -I.. $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*

