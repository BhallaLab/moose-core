include_directories(../basecode)
add_library(biophysics 
	SpikeRingBuffer.cpp	
	SynHandler.cpp	
	IntFire.cpp	
	Synapse.cpp	
	SpikeGen.cpp	
	CompartmentDataHolder.cpp	
	CompartmentBase.cpp	
	Compartment.cpp	
	SymCompartment.cpp	
	ChanBase.cpp	
	HHChannel.cpp	
	HHGate.cpp	
	HHGate2D.cpp	
	HHChannel2D.cpp	
        CaConcBase.cpp
	CaConc.cpp	
	MgBlock.cpp	
	Neuron.cpp	
	ReadCell.cpp	
	SynChanBase.cpp	
	SynChan.cpp	
	testBiophysics.cpp	
	IzhikevichNrn.cpp	
	DifShell.cpp	
	Leakage.cpp	
	VectorTable.cpp	
	MarkovRateTable.cpp	
	MarkovChannel.cpp	
	MarkovGslSolver.cpp	
	MatrixOps.cpp	
	MarkovSolverBase.cpp	
	MarkovSolver.cpp	
	VClamp.cpp	
	# LeakyIaF.cpp	
	# GapJunction.cpp	
	Nernst.cpp	
	# GHK.cpp		
	# NMDAChan.cpp	
    )
