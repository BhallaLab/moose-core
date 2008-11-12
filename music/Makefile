#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = music.o

OBJ = \
	InputEventPort.o	\
	InputEventChannel.o	\
	OutputEventPort.o	\
	OutputEventChannel.o	\
	Music.o	

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h 

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Music.o : Music.h 
InputEventChannel.o : InputEventChannel.h 
InputEventPort.o : InputEventPort.h InputEventChannel.h 
OutputEventChannel.o : OutputEventChannel.h 
OutputEventPort.o : OutputEventPort.h OutputEventChannel.h 

.cpp.o:
	$(CXX) $(CXXFLAGS) -I../basecode -I../randnum -I.. $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)
#	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
