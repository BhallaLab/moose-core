cmake_minimum_required(VERSION 2.6)

include_directories(../msg)
include_directories(../basecode)
add_library(benchmarks 
    benchmarks.cpp
    kineticMarks.cpp
    )
target_link_libraries(benchmarks shell)
