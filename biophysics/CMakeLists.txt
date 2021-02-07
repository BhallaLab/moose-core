cmake_minimum_required(VERSION 3.12)

set(BIOPHYSICS_SRCS
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
    SwcSegment.cpp
    ReadSwc.cpp
    SynChan.cpp
    NMDAChan.cpp
    IzhikevichNrn.cpp
    DifShellBase.cpp
    DifShell.cpp
    DifBufferBase.cpp
    DifBuffer.cpp
    MMPump.cpp
    Leakage.cpp
    VectorTable.cpp
    MarkovRateTable.cpp
    MarkovChannel.cpp
    MatrixOps.cpp
    MarkovSolverBase.cpp
    MarkovSolver.cpp
    VClamp.cpp
    Spine.cpp
    MarkovOdeSolver.cpp
    testBiophysics.cpp
    )

add_library(biophysics ${BIOPHYSICS_SRCS})

target_include_directories(biophysics PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/odeint-v2/include)

