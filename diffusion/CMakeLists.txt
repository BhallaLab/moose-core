cmake_minimum_required(VERSION 2.8)
include_directories(../basecode ../utility ../ksolve)
add_library(diffusion
	FastMatrixElim.cpp
	DiffPoolVec.cpp
	Dsolve.cpp
        testDiffusion.cpp
    )
target_link_libraries(diffusion ksolve basecode)
