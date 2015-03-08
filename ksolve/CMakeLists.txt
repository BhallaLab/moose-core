cmake_minimum_required(VERSION 2.8)
include_directories(../basecode ../utility ../kinetics ../external/muparser)
add_library(ksolve
	KinSparseMatrix.cpp	
	ZombiePool.cpp 
        ZombieBufPool.cpp 
        #ZombieFuncPool.cpp 
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
if(USE_GSL)
    target_link_libraries(ksolve gsl gslcblas)
endif()
