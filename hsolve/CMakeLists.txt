cmake_minimum_required(VERSION 2.8)
include_directories(../basecode ../utility ../kinetics ../external/debug)


add_library(hsolve
    	HSolveStruct.cpp 
	HinesMatrix.cpp 
	HSolvePassive.cpp 
	RateLookup.cpp 
	HSolveActive.cpp 
	HSolveActiveSetup.cpp 
	HSolveInterface.cpp 
	HSolve.cpp 
	HSolveUtils.cpp 
	testHSolve.cpp 
	ZombieCompartment.cpp 
	ZombieCaConc.cpp 
	ZombieHHChannel.cpp 
)

if(ENABLE_CUDA)
    FILE(GLOB CUDA_SRC *.cu)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    CUDA_COMPILE(HSOLVE_CUDA_OBJS ${CUDA_SRC})
    CUDA_ADD_LIBRARY(hsolve_cuda ${HSOLVE_CUDA_OBJS})
    TARGET_LINK_LIBRARIES(hsolve_cuda ${CUDA_LIBRARIES})
    TARGET_LINK_LIBRARIES(hsolve hsolve_cuda)
endif(ENABLE_CUDA)



