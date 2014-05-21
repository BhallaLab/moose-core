cmake_minimum_required(VERSION 2.8)
include_directories(../basecode ../utility ../kinetics)
add_library(signeur
    Adaptor.cpp
    testSigNeur.cpp
    )
target_link_libraries(signeur mesh biophysics)
