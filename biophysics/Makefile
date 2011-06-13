#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = biophysics.o

OBJ = \
	SynBase.o	\
	IntFire.o	\
	Synapse.o	\
	Compartment.o	\
	SymCompartment.o	\
	ChanBase.o	\
	SynChanBase.o	\
	HHChannel.o	\
	HHGate.o	\
	SynChan.o	\
	SpikeGen.o	\
	CaConc.o	\
	Nernst.o	\
	GHK.o		\
	NMDAChan.o	\
	HHGate2D.o	\
	HHChannel2D.o	\
	MgBlock.o	\
	VectorTable.o \
	MarkovGsl.o \
	MarkovRateTable.o \
	MarkovChannel.o \
	testBiophysics.o	\

GSL_LIBS = -L/usr/lib -lgsl

HEADERS = \
	../basecode/header.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
SynBase.o:	SynBase.h Synapse.h 
IntFire.o:	IntFire.h SynBase.h Synapse.h 
Synapse.o:	SynBase.h Synapse.h
Compartment.o: Compartment.h
SymCompartment.o: Compartment.h SymCompartment.h
ChanBase.o: ChanBase.h
HHChannel.o: ChanBase.h HHChannel.h HHGate.h
HHGate.o: HHGate.h
SynChanBase.o: Synapse.h SynBase.h ChanBase.h
SynChan.o: SynBase.h ChanBase.h Synapse.h SynChan.h
NMDAChan.o: NMDAChan.h SynBase.h ChanBase.h Synapse.h SynChan.h
SpikeGen.o: SpikeGen.h
CaConc.o:	CaConc.h
Nernst.o:	Nernst.h
MgBlock.o:	MgBlock.h
GHK.o:	GHK.h
HHGate2D.o:	../builtins/Interpol2D.h HHGate2D.h
HHChannel2D.o: ChanBase.h HHChannel2D.h HHGate2D.h
VectorTable.o : VectorTable.h
MarkovGsl.o : MarkovGsl.h
MarkovRateTable.o : VectorTable.h ../builtins/Interpol2D.h MarkovRateTable.h 
MarkovChannel.o : VectorTable.h ../builtins/Interpol2D.h MarkovRateTable.h MarkovGsl.h ChanBase.h MarkovChannel.h 
testBiophysics.o: SynBase.h IntFire.h Synapse.h Compartment.h HHChannel.h HHGate.h \
 									VectorTable.h MarkovRateTable.h	../builtins/Interpol2D.h \
								 	MarkovGsl.h MarkovChannel.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg $< -c

$(TARGET): $(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) $(GSL_LIBS)

clean:
	rm -f *.o $(TARGET) core core.*
