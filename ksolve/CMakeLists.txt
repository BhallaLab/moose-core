cmake_minimum_required(VERSION 2.6)
include(CheckIncludeFileCXX)

include_directories(../builtins ../basecode ../utility ../kinetics)
include_directories(../ )
include_directories(../mesh)
include_directories(../external/muparser/include )
IF(WITH_BOOST)
    check_include_file_cxx( ${Boost_INCLUDE_DIRS}/boost/numeric/odeint.hpp
        ODEINT_EXISTS)
    # If boost distribution does not have its own odeint library, use the
    # private one.
    if(NOT ODEINT_EXISTS)
        include_directories( ../external/odeint-v2/include )
    endif(NOT ODEINT_EXISTS)
    include_directories( ../external/boost-numeric-bindings )
elseif(WITH_GSL)
    include_directories( ${GSL_INCLUDE_DIRS} )
endif(WITH_BOOST)

add_library(ksolve
	KinSparseMatrix.cpp	
	ZombiePool.cpp 
        ZombieFunction.cpp
        ZombieBufPool.cpp 
	ZombieReac.cpp 
	ZombieEnz.cpp 
	ZombieMMenz.cpp 
        VoxelPoolsBase.cpp
	VoxelPools.cpp 
        BoostSys.cpp
        GssaVoxelPools.cpp
	RateTerm.cpp 
        FuncTerm.cpp
	Stoich.cpp 
	Ksolve.cpp 
        SteadyState.cpp
        Gsolve.cpp
        ZombiePoolInterface.cpp
        testKsolve.cpp
    )
