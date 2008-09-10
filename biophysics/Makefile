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
	TestBiophysics.o	\
	BinSynchan.o	\
	StochSynchan.o	\
	PulseGen.o	\
	RandomSpike.o	
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
	BinSynchan.o	\
	StochSynchan.o	\
	PulseGen.o	\
	RandomSpike.o \
	Cell.o		\
	BioScan.o	\
	TestBiophysics.o \
	script_out.o
endif

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
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
script_out.o:	script_out.h

ifneq (,$(findstring -DCRL_MPI,$(CFLAGS)))
ParSpikeGen.o:	SpikeGen.h ParSpikeGen.h
endif

SynChan.o:	SynChan.h SynInfo.h

ifneq (,$(findstring -DCRL_MPI,$(CFLAGS)))
ParSynChan.o:	SynChan.h SynInfo.h ParSynChan.h
endif

Cell.o:	Cell.h ../element/Neutral.h
BioScan.o:		BioScan.h
StochSynchan.o:	StochSynchan.h StochSynInfo.h
BinSynchan.o:	BinSynchan.h BinSynInfo.h ../randnum/BinomialRng.h
.cpp.o:
	$(CXX) $(CFLAGS) -I../basecode -I../randnum -I.. $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)
#	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
