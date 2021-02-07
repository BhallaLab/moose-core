cmake_minimum_required(VERSION 3.12)

set(KSOLVE_SRCS
    KinSparseMatrix.cpp
    ZombiePool.cpp
    ZombieFunction.cpp
    ZombieBufPool.cpp
    ZombieReac.cpp
    ZombieEnz.cpp
    ZombieMMenz.cpp
    VoxelPoolsBase.cpp
    VoxelPools.cpp
    GssaVoxelPools.cpp
    RateTerm.cpp
    FuncTerm.cpp
    Stoich.cpp
    Ksolve.cpp
    Gsolve.cpp
    ZombiePoolInterface.cpp
    SteadyState.cpp
    testKsolve.cpp
    )

add_library( ksolve ${KSOLVE_SRCS} )

target_include_directories(ksolve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../external/eigen-3.3.8)

