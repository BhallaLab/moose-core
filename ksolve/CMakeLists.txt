cmake_minimum_required(VERSION 2.8)
include(CheckIncludeFileCXX)

include_directories(../builtins ../basecode ../utility ../kinetics)
include_directories(../ )
include_directories(../mesh)
include_directories(../external/muparser/include )

find_package(Threads REQUIRED)

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

if(PARALLELIZED_SOLVERS)
    message( STATUS "Parallel version of KSolve and Gsolve" )
    add_definitions( -DPARALLELIZE_KSOLVE_WITH_CPP11_ASYNC )
    add_definitions( -DPARALLELIZE_GSOLVE_WITH_CPP11_ASYNC )
else()
    message( STATUS "Serial version of KSolve and Gsolve" )
endif(PARALLELIZED_SOLVERS)

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
        testKsolve.cpp
    )

if(WITH_GSL)
    list(APPEND KSOLVE_SRCS SteadyStateGsl.cpp )
elseif(WITH_BOOST)
    list(APPEND KSOLVE_SRCS SteadyStateBoost.cpp )
    list(APPEND KSOLVE_SRCS BoostSys.cpp)
endif(WITH_GSL)

add_library( ksolve ${KSOLVE_SRCS} )
add_library( lsoda ${CMAKE_CURRENT_SOURCE_DIR}/LSODA.cpp)

if( "${CMAKE_BUILD_TYPE}" STREQUAL "Release" )
    set_target_properties( ksolve lsoda PROPERTIES COMPILE_FLAGS "-Werror -Wall")
endif()

add_executable( test_lsoda ${CMAKE_CURRENT_SOURCE_DIR}/test_LSODA.cpp)
target_link_libraries( test_lsoda lsoda )

add_executable( benchmark_lsoda ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_LSODA.cpp)
target_link_libraries( benchmark_lsoda lsoda ${CMAKE_THREAD_LIBS_INIT})
set_target_properties( benchmark_lsoda PROPERTIES COMPILE_FLAGS "-O3 -pthread" LINK_FLAGS "-pthread")

enable_testing()
add_test(NAME test_lsoda COMMAND $<TARGET_FILE:test_lsoda>)
