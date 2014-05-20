CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
add_library(simpletest STATIC simple_test.hpp simple_assert.hpp)
set_target_properties(simpletest PROPERTIES LINKER_LANGUAGE CXX)
