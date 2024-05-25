cmake_minimum_required(VERSION 3.20)


set( SYNAPSE_SRCS
    GraupnerBrunel2012CaPlasticitySynHandler.cpp
    RollingMatrix.cpp
    SeqSynHandler.cpp
    SimpleSynHandler.cpp
    STDPSynapse.cpp
    STDPSynHandler.cpp
    Synapse.cpp
    SynHandlerBase.cpp
    testSynapse.cpp
    )

add_library(synapse ${SYNAPSE_SRCS} )
