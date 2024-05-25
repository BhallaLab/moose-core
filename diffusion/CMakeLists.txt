cmake_minimum_required(VERSION 3.20)


if(WITH_GSL)
   include_directories(${GSL_INCLUDE_DIRS})
endif(WITH_GSL)

add_library(diffusion
	FastMatrixElim.cpp
	DiffPoolVec.cpp
	Dsolve.cpp
        testDiffusion.cpp
    )
