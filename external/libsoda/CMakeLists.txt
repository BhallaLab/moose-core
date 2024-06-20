cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

add_library(lsoda ${CMAKE_CURRENT_SOURCE_DIR}/LSODA.cpp)


# MSVC linker throws error when dll and static libs are mixed
# TODO find workaround
if(NOT MSVC)
       add_executable(cpp_test_lsoda test_LSODA.cpp)
       target_link_libraries(cpp_test_lsoda lsoda)

       add_executable(cpp_benchmark_lsoda benchmark_LSODA.cpp)
       target_link_libraries(cpp_benchmark_lsoda lsoda ${CMAKE_THREAD_LIBS_INIT})

       enable_testing()
       add_test(NAME cpp_test_lsoda COMMAND $<TARGET_FILE:cpp_test_lsoda>)
       add_test(NAME cpp_test_benchmark COMMAND $<TARGET_FILE:cpp_benchmark_lsoda>)
endif()
