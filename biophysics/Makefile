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

ifneq (,$(findstring -DCRL_MPI,$(CFLAGS)))
OBJ = \
	Compartment.o	\
	SymCompartment.o	\
	HHChannel.o	\
	Mg_block.o	\
	HHGate.o	\
	CaConc.o	\
	Nernst.o	\
	SpikeGen.o	\
	ParSpikeGen.o	\
	SynChan.o	\
	ParSynChan.o	\
	HSolveStruct.o	\
	HSolveBase.o	\
	HSolve.o \
	NeuroScanBase.o \
	NeuroScan.o \
	NeuroHub.o \
	Cell.o	\
	TestBiophysics.o	\
	BinSynchan.o	\
	StochSynchan.o	
else
OBJ = \
	Compartment.o	\
	SymCompartment.o	\
	HHChannel.o	\
	Mg_block.o	\
	HHGate.o	\
	CaConc.o	\
	Nernst.o	\
	SpikeGen.o	\
	SynChan.o	\
	HSolveStruct.o	\
	HSolveBase.o	\
	HSolve.o \
	NeuroScanBase.o \
	NeuroScan.o \
	NeuroHub.o \
	Cell.o	\
	TestBiophysics.o	\
	BinSynchan.o	\
	StochSynchan.o	
endif

HEADERS = \
	../basecode/header.h \
	../basecode/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Compartment.o:	Compartment.h ../basecode/Ftype2.h
SymCompartment.o:	Compartment.h SymCompartment.h ../basecode/Ftype2.h
HHChannel.o:	HHChannel.h ../basecode/Ftype2.h ../basecode/Ftype3.h
Mg_block.o:	Mg_block.h ../basecode/Ftype2.h ../basecode/Ftype3.h
HHGate.o:	HHGate.h ../builtins/Interpol.h
CaConc.o:	CaConc.h
Nernst.o:	Nernst.h
SpikeGen.o:	SpikeGen.h

ifneq (,$(findstring -DCRL_MPI,$(CFLAGS)))
ParSpikeGen.o:	SpikeGen.h ParSpikeGen.h
endif

SynChan.o:	SynChan.h SynInfo.h

ifneq (,$(findstring -DCRL_MPI,$(CFLAGS)))
ParSynChan.o:	SynChan.h SynInfo.h ParSynChan.h
endif

HSolveStruct.o:	HSolveStruct.h
HSolveBase.o:	HSolveBase.h HSolveStruct.h
HSolve.o:	HSolve.h HSolveBase.h HSolveStruct.h
NeuroScanBase.o:	NeuroScanBase.h HSolveStruct.h
NeuroScan.o:	NeuroScan.h NeuroScanBase.h HSolveStruct.h
NeuroHub.o:	NeuroHub.h HSolveStruct.h
Cell.o:		Cell.h 
StochSynchan.o:	StochSynchan.h StochSynInfo.h
BinSynchan.o:	BinSynchan.h BinSynInfo.h ../utility/randnum/BinomialRng.h
.cpp.o:
	$(CXX) $(CFLAGS) -I../basecode -I../utility/randnum -I.. $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)
#	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
