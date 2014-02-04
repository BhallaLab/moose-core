#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU Lesser General Public License version 2.1
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = _biophysics.o

OBJ = \
	SpikeRingBuffer.o	\
	SynHandler.o	\
	IntFire.o	\
	Synapse.o	\
	SpikeGen.o	\
	Compartment.o	\
	SymCompartment.o	\
	ChanBase.o	\
	HHChannel.o	\
	HHGate.o	\
	CaConc.o	\
	Neuron.o	\
	ReadCell.o	\
	testBiophysics.o	\

# IzhikevichNrn.o	\
	LeakyIaF.o	\
	GapJunction.o	\
	Nernst.o	\
	GHK.o		\
	NMDAChan.o	\
	HHGate2D.o	\
	HHChannel2D.o	\
	MgBlock.o	\
	VectorTable.o	\
	MarkovRateTable.o	\
	MarkovChannel.o	\
	MarkovGslSolver.o	\
	MatrixOps.o	\
	MarkovSolverBase.o	\
	MarkovSolver.o	\
	VClamp.o	\

# GSL_LIBS = -L/usr/lib -lgsl

HEADERS = \
	../basecode/header.h \
	../utility/numutil.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
SpikeRingBuffer.o:	SpikeRingBuffer.h
SynHandler.o:	SpikeRingBuffer.h SynHandler.h Synapse.h 
IntFire.o:	IntFire.h SpikeRingBuffer.h SynHandler.h Synapse.h 
SpikeGen.o: SpikeGen.h
Compartment.o: Compartment.h
SymCompartment.o: Compartment.h SymCompartment.h
ChanBase.o: ChanBase.h
HHChannel.o: ChanBase.h HHChannel.h HHGate.h
HHGate.o: HHGate.h
CaConc.o:	CaConc.h
Neuron.o:	Neuron.h
ReadCell.o: ReadCell.h ../shell/Shell.h ../utility/utility.h
#IzhikevichNrn.o: IzhikevichNrn.h
#LeakyIaF.o: LeakyIaF.h
#Synapse.o:	SynBase.h Synapse.h
#GapJunction.o: GapJunction.h
#NMDAChan.o: NMDAChan.h SynBase.h ChanBase.h Synapse.h SynChan.h
#Nernst.o:	Nernst.h
#MgBlock.o:	MgBlock.h
#GHK.o:	GHK.h
#HHGate2D.o:	../builtins/Interpol2D.h HHGate2D.h
#HHChannel2D.o: ChanBase.h HHChannel2D.h HHGate2D.h
#VectorTable.o : VectorTable.h
#MarkovGslSolver.o : MarkovGslSolver.h 
#MatrixOps.o : MatrixOps.h
#MarkovRateTable.o : VectorTable.h ../builtins/Interpol2D.h MarkovRateTable.h 
#MarkovSolverBase.o : MarkovSolver.h MatrixOps.h MarkovRateTable.h
#MarkovSolver.o : MarkovSolverBase.h MarkovSolver.h
#MarkovChannel.o : VectorTable.h ../builtins/Interpol2D.h MarkovRateTable.h ChanBase.h MarkovChannel.h 
#VClamp.o: VClamp.h
#testBiophysics.o: SynBase.h IntFire.h Synapse.h Compartment.h HHChannel.h HHGate.h 

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg $< -c

$(TARGET): $(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) $(GSL_LIBS)

clean:
	rm -f *.o $(TARGET) core core.*
