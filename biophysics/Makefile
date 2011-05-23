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
	SynHandler.o	\
	IntFire.o	\
	Synapse.o	\
	Compartment.o	\
	SymCompartment.o	\
	HHChannel.o	\
	HHGate.o	\
	SynChan.o	\
	SpikeGen.o	\
	CaConc.o	\
	Nernst.o	\
	testBiophysics.o	\

HEADERS = \
	../basecode/header.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
SynHandler.o:	SynHandler.h Synapse.h 
IntFire.o:	IntFire.h SynHandler.h Synapse.h 
Synapse.o:	SynHandler.h Synapse.h
Compartment.o: Compartment.h
SymCompartment.o: Compartment.h SymCompartment.h
HHChannel.o: HHChannel.h HHGate.h
HHGate.o: HHGate.h
SynChan.o: SynHandler.h Synapse.h SynChan.h
SpikeGen.o: SpikeGen.h
CaConc.o:	CaConc.h
Nernst.o:	Nernst.h
testBiophysics.o: SynHandler.h IntFire.h Synapse.h Compartment.h HHChannel.h HHGate.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
