#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = robots.o

OBJ = \
	SigNeur.o	\
	SigNeurNeuron.o	\
	SigNeurSignaling.o	\
	SigNeurDiffusion.o	\
	SigNeurAdaptors.o	\
	Adaptor.o	\

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
SigNeur.o:	SigNeur.h
SigNeurNeuron.o:	SigNeur.h
SigNeurSignaling.o:	SigNeur.h
SigNeurDiffusion.o:	SigNeur.h
SigNeurAdaptors.o:	SigNeur.h
Adaptor.o:	Adaptor.h

.cpp.o:
	$(CXX) $(CXXFLAGS) -I.. -I../basecode $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
