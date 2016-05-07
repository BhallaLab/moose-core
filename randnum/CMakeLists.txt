file(GLOB random_SRC "*.cpp")
include_directories(../basecode .. ${GSL_INCLUDE_DIRS})
add_library(randnum ${random_SRC})
