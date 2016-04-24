cmake_minimum_required(VERSION 2.6)
include_directories(../builtins ../basecode ../utility ../kinetics)
include_directories(../)
include_directories(../mesh)
include_directories(../external/muparser/include )

if(USE_GSL)
    include_directories( ${GSL_INCLUDE_DIRS} )
endif(USE_GSL)

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
        GssaVoxelPools.cpp
	RateTerm.cpp 
        FuncTerm.cpp
	Stoich.cpp 
	Ksolve.cpp 
        SteadyState.cpp
        Gsolve.cpp
        ZombiePoolInterface.cpp
        BoostSys.cpp
        testKsolve.cpp
    )
