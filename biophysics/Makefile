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
	SymCompartment.o	\
	HHChannel.o	\
	HHChannel2D.o	\
	Mg_block.o	\
	HHGate.o	\
	HHGate2D.o	\
	CaConc.o	\
	Nernst.o	\
	SpikeGen.o	\
	SynChan.o	\
	BinSynchan.o	\
	StochSynchan.o	\
	STPSynChan.o	\
	NMDAChan.o	\
	RandomSpike.o \
	Cell.o		\
	BioScan.o	\
	TestBiophysics.o \
	script_out.o \
	DifShell.o	\
	TauPump.o	\
	GHK.o \
	IntFire.o	\
	IzhikevichNrn.o \
	Leakage.o	\
	KinSynChan.o	\
	StochSpikeGen.o \



HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Compartment.o:	Compartment.h ../basecode/Ftype2.h
SymCompartment.o:	Compartment.h SymCompartment.h ../basecode/Ftype2.h
HHChannel.o:	HHChannel.h ../basecode/Ftype2.h ../basecode/Ftype3.h
HHChannel2D.o:	HHChannel.h HHChannel2D.h ../basecode/Ftype2.h ../basecode/Ftype3.h
Mg_block.o:	Mg_block.h ../basecode/Ftype2.h ../basecode/Ftype3.h
HHGate.o:	HHGate.h ../builtins/Interpol.h
HHGate2D.o:	HHGate.h HHGate2D.h ../builtins/Interpol.h ../builtins/Interpol2D.h
CaConc.o:	CaConc.h
Nernst.o:	Nernst.h
SpikeGen.o:	SpikeGen.h
script_out.o:	script_out.h
SynChan.o:	SynChan.h SynInfo.h
STPSynChan.o:	STPSynChan.h SynChan.h SynInfo.h
NMDAChan.o:	NMDAChan.h SynInfo.h
Cell.o:	Cell.h ../element/Neutral.h
BioScan.o:		BioScan.h
StochSynchan.o:	StochSynchan.h StochSynInfo.h
BinSynchan.o:	BinSynchan.h BinSynInfo.h ../randnum/BinomialRng.h
DifShell.o: DifShell.h
TauPump.o: TauPump.h
GHK.o: GHK.h ../basecode/Ftype2.h
IntFire.o: IntFire.h
IzhikevichNrn.o: IzhikevichNrn.h
Leakage.o: Leakage.h
KinSynChan.o: KinSynChan.h SynChan.h SynInfo.h
StochSpikeGen.o: SpikeGen.h StochSpikeGen.h ../randnum/mt19937ar.cpp

.cpp.o:
	$(CXX) $(CXXFLAGS) -I../basecode -I../randnum -I.. $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)
#	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
