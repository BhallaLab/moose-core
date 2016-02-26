cmake_minimum_required(VERSION 2.8)
include_directories(../basecode ../utility ../kinetics ../external/debug)

option(DEBUG_CUDA "Enable CUDA debug macros" OFF)
if(DEBUG_CUDA)
    add_definitions(-DDEBUG_CUDA)
endif()

if(ENABLE_CUDA)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
    SET(CMAKE_CXX_FLAGS -fPIC)
    FILE(GLOB CUDA_SRC *.cu)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    CUDA_COMPILE(HSOLVE_CUDA_OBJS ${CUDA_SRC})
    CUDA_ADD_LIBRARY(hsolve_cuda ${HSOLVE_CUDA_OBJS})
    TARGET_LINK_LIBRARIES(hsolve_cuda ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY})
endif(ENABLE_CUDA)


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
    #RateLookup.cpp
    testHSolve.cpp
    ZombieCaConc.cpp
    ZombieCompartment.cpp
    ZombieHHChannel.cpp
    )
if(ENABLE_CUDA)
    TARGET_LINK_LIBRARIES(hsolve hsolve_cuda)
endif(ENABLE_CUDA)



