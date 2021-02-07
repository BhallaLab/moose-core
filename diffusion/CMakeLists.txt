cmake_minimum_required(VERSION 3.12)

add_library(diffusion
    FastMatrixElim.cpp
    DiffPoolVec.cpp
    Dsolve.cpp
    testDiffusion.cpp
    )
