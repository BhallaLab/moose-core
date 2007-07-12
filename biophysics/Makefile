#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = biophysics.o

OBJ = \
	Compartment.o	\
	HHChannel.o	\
	HHGate.o	\
	CaConc.o	\
	Nernst.o	\
	SpikeGen.o	\
	SynChan.o	\
	HSolveBase.o	\
	HSolve.o \
	NeuroScanBase.o \
	NeuroScan.o \
	NeuroHub.o \
	TestBiophysics.o	\

HEADERS = \
	../basecode/header.h \
	../basecode/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Compartment.o:	Compartment.h ../basecode/Ftype2.h
HHChannel.o:	HHChannel.h ../basecode/Ftype2.h ../basecode/Ftype3.h
HHGate.o:	HHGate.h ../builtins/Interpol.h
CaConc.o:	CaConc.h
Nernst.o:	Nernst.h
SpikeGen.o:	SpikeGen.h
SynChan.o:	SynChan.h SynInfo.h
HSolveBase.o:	HSolveBase.h HSolveStructure.h
HSolve.o:	HSolve.h HSolveBase.h HSolveStructure.h
NeuroScanBase.o:	NeuroScanBase.h HSolveStructure.h
NeuroScan.o:	NeuroScan.h NeuroScanBase.h HSolveStructure.h
NeuroHub.o:	NeuroHub.h HSolveStructure.h

.cpp.o:
	$(CXX) $(CFLAGS) -I../basecode $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
