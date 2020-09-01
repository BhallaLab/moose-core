cmake_minimum_required(VERSION 2.8)
include( ${CMAKE_CURRENT_SOURCE_DIR}/../CheckCXXCompiler.cmake)

add_library(utility
    strutil.cpp
    types.cpp
    setupenv.cpp
    numutil.cpp
    Annotator.cpp
    Vec.cpp
    cnpy.cpp
    fileutils.cpp
    utility.cpp
    )
target_link_libraries(utility PRIVATE fmt)

add_executable(test_cnpy test_cnpy.cpp)

target_link_libraries(test_cnpy utility)

set(TEST_SCRIPT ${CMAKE_BINARY_DIR}/test_cnpy.py)
file(WRITE ${TEST_SCRIPT} "import numpy as np")

enable_testing()
add_test(NAME cpp_test_cnpy COMMAND $<TARGET_FILE:test_cnpy>)

add_executable(test_util test_util.cpp)
target_link_libraries(test_util utility)

enable_testing()
add_test(NAME cpp_test_util  COMMAND $<TARGET_FILE:test_util>)
