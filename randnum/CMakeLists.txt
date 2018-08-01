cmake_minimum_required(VERSION 2.8)

file(GLOB random_SRC "*.cpp")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/basecode
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
if(WITH_GSL)
    include_directories(${GSL_INCLUDE_DIRS})
endif(WITH_GSL)

add_library(randnum ${random_SRC})

add_executable(test_randnum test_randnum.cpp ../basecode/global.cpp)
target_link_libraries( test_randnum randnum )

enable_testing()
