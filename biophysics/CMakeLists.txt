include_directories(../basecode ../synapse ../utility ../)
include_directories(../external/muparser/include )
if(WITH_GSL)
    include_directories(${GSL_INCLUDE_DIRS})
endif(WITH_GSL)

set(BIOPHYSICS_SRC 
    CaConcBase.cpp
    CaConc.cpp
    ChanBase.cpp
    ChanCommon.cpp
    CompartmentBase.cpp
    Compartment.cpp
    CompartmentDataHolder.cpp
    DifShell.cpp
    GapJunction.cpp
    HHChannel2D.cpp
    HHChannelBase.cpp
    HHChannel.cpp
    HHGate2D.cpp
    HHGate.cpp
    IntFire.cpp
    IzhikevichNrn.cpp
    Leakage.cpp
    MarkovChannel.cpp
    MarkovRateTable.cpp
    MarkovSolverBase.cpp
    MarkovSolver.cpp
    MatrixOps.cpp
    MgBlock.cpp
    Nernst.cpp
    Neuron.cpp
    NMDAChan.cpp
    RandSpike.cpp
    ReadCell.cpp
    ReadSwc.cpp
    SpikeGen.cpp
    SpikeRingBuffer.cpp
    Spine.cpp
    SwcSegment.cpp
    SymCompartment.cpp
    SynChan.cpp
    testBiophysics.cpp
    VClamp.cpp
    VectorTable.cpp
    )

if(USE_GSL)
    list(APPEND BIOPHYSICS_SRC MarkovGslSolver.cpp)
endif(USE_GSL)

add_library(biophysics ${BIOPHYSICS_SRC} )
