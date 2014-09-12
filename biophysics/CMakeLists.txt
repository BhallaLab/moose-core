include_directories(../basecode ../synapse)
add_library(biophysics 
	IntFire.cpp	
	SpikeGen.cpp	
        RandSpike.cpp
	CompartmentDataHolder.cpp	
	CompartmentBase.cpp	
	Compartment.cpp	
	SymCompartment.cpp	
        GapJunction.cpp
	ChanBase.cpp	
        ChanCommon.cpp
	HHChannel.cpp	
        HHChannelBase.cpp
        HHChannel2D.cpp
	HHGate.cpp	
	HHGate2D.cpp	
	HHChannel2D.cpp	
        CaConcBase.cpp
	CaConc.cpp	
	MgBlock.cpp	
        Nernst.cpp
	Neuron.cpp	
	ReadCell.cpp	
	#SynChanBase.cpp	
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
	# GHK.cpp		
	# NMDAChan.cpp	
        testBiophysics.cpp
    )
