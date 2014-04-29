cmake_minimum_required(VERSION 2.6)

file(GLOB files_SRC 
    "*.h"
    "*.cpp"
)
include_directories(../msg)
include_directories(../basecode)
add_library(benchmarks ${files_SRC})
target_link_libraries(benchmarks shell)
