cmake_minimum_required(VERSION 3.20)

add_library(randnum RNG.cpp randnum.cpp)

add_executable(normal_dist ${CMAKE_CURRENT_SOURCE_DIR}/test_normal_dist.cpp)

if(NOT MSVC)
enable_testing()
add_test(NAME cpp_test_normal_dist COMMAND $<TARGET_FILE:normal_dist> )
endif()
