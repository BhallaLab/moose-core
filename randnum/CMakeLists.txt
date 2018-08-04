cmake_minimum_required(VERSION 2.8)
file(GLOB random_SRC "*.cpp")
add_library(randnum ${random_SRC})
