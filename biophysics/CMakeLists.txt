include_directories(../basecode)
add_library(biophysics 
    CaConc.cpp
    ChanBase.cpp
    Compartment.cpp
    HHChannel.cpp
    HHGate.cpp
    IntFire.cpp
    Neuron.cpp
    ReadCell.cpp
    SpikeGen.cpp
    SpikeRingBuffer.cpp
    SymCompartment.cpp
    Synapse.cpp
    SynChanBase.cpp
    SynChan.cpp
    SynHandler.cpp
    testBiophysics.cpp
    )

target_link_libraries(biophysics debug_print)
