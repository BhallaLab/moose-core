file(GLOB scheduling_SRC  "*.cpp" )

if(NOTIFY_PROGRESS)
    add_definitions("-DNOTIFY_PROGRESS")
endif(NOTIFY_PROGRESS)

if(PARALLELIZED_CLOCK)
    message( STATUS "Enabling high-level parallelization of moose::Clock" )
    add_definitions( -DPARALLELIZE_CLOCK_USING_CPP11_ASYNC )
endif(PARALLELIZED_CLOCK)

include_directories(../msg)
include_directories(../basecode)

add_library(scheduling ${scheduling_SRC})

if(WITH_CUDA)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )
    FILE(GLOB CUDA_SRC *.cu)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    CUDA_COMPILE(SCHEDULING_CUDA_OBJS ${CUDA_SRC})
    CUDA_ADD_LIBRARY(scheduling_cuda ${SCHEDULING_CUDA_OBJS})
    TARGET_LINK_LIBRARIES(scheduling_cuda 
        ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} 
        )
endif()

if(WITH_CUDA)
    target_link_libraries( scheduling scheduling_cuda )
endif(WITH_CUDA)
