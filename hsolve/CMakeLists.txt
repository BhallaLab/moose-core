cmake_minimum_required(VERSION 2.8)
include_directories(../basecode ../utility ../kinetics ../external/debug)
add_library(hsolve
    Cell.cpp
    HinesMatrix.cpp
    HSolveActive.cpp
    HSolveActiveSetup.cpp
    HSolve.cpp
    #HSolveHub.cpp
    HSolveInterface.cpp
    HSolvePassive.cpp
    HSolveStruct.cpp
    HSolveUtils.cpp
    RateLookup.cpp
    testHSolve.cpp
    ZombieCaConc.cpp
    ZombieCompartment.cpp
    ZombieHHChannel.cpp
    )

target_link_libraries(hsolve basecode)
