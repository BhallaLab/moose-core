include_directories(../external/cuda)
include_directories(../msg)
include_directories(../basecode)

option(DEBUG_CUDA "Enable CUDA debug macros" OFF)
if(DEBUG_CUDA)
    add_definitions(-DDEBUG_CUDA)
endif()

if(WITH_CUDA)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )
    FILE(GLOB CUDA_SRC *.cu)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    CUDA_COMPILE(KSOLVE_CUDA_OBJS ${CUDA_SRC})
    CUDA_ADD_LIBRARY(scheduling_cuda ${KSOLVE_CUDA_OBJS})
    TARGET_LINK_LIBRARIES(scheduling_cuda 
        ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} 
        )
endif()

if(NOT WITH_CUDA )
    set(scheduling_SRC "Clock.cpp" )
endif(NOT WITH_CUDA)

list(APPEND scheduling_SRC CudaKsolve.cpp testScheduling.cpp)

if(NOTIFY_PROGRESS)
    ADD_DEFINITIONS("-DNOTIFY_PROGRESS")
endif()

add_library(scheduling ${scheduling_SRC})

if(WITH_CUDA)
    target_link_libraries( scheduling scheduling_cuda )
endif(WITH_CUDA)
