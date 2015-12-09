cmake_minimum_required(VERSION 2.6)
include_directories(../builtins ../basecode ../utility ../kinetics ../external/muparser)
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
        testKsolve.cpp
    )

