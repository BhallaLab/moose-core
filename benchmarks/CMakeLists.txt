cmake_minimum_required(VERSION 2.8)

include_directories(../msg)
include_directories(../basecode)
add_library(benchmarks 
    benchmarks.cpp
    kineticMarks.cpp
    )

# This target run benchmarks. Not a part of moose distribution
add_executable( 
    randnum_benchmark 
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_random_numbers.cpp
    )
set_target_properties( randnum_benchmark  
    PROPERTIES COMPILE_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11"
    )
target_link_libraries(randnum_benchmark randnum  )
