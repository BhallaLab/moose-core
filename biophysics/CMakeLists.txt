include_directories(../basecode)
add_library(biophysics 
	SpikeRingBuffer.cpp	
	SynHandler.cpp	
	IntFire.cpp	
	Synapse.cpp	
	SpikeGen.cpp	
	Compartment.cpp	
	SymCompartment.cpp	
	ChanBase.cpp	
	HHChannel.cpp	
	HHGate.cpp	
	CaConc.cpp	
	Neuron.cpp	
	ReadCell.cpp	
	SynChanBase.cpp	
	SynChan.cpp	
	testBiophysics.cpp	
	IzhikevichNrn.cpp	
    )

target_link_libraries(biophysics debug_print)
