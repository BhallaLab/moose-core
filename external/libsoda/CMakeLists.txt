cmake_minimum_required(VERSION 2.8)

add_library(lsoda ${CMAKE_CURRENT_SOURCE_DIR}/LSODA.cpp)

find_package(Threads REQUIRED)

add_executable(test_lsoda test_LSODA.cpp)
target_link_libraries(test_lsoda lsoda)

add_executable(benchmark_lsoda benchmark_LSODA.cpp)
target_link_libraries(benchmark_lsoda lsoda)
target_link_libraries(benchmark_lsoda ${CMAKE_THREAD_LIBS_INIT})

enable_testing()
add_test(NAME test_lsoda COMMAND $<TARGET_FILE:test_lsoda>)
add_test(NAME test_benchmark COMMAND $<TARGET_FILE:benchmark_lsoda>)
