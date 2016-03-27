cmake_minimum_required(VERSION 2.8)
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/.. )
add_library( randnum 
    ${CMAKE_CURRENT_SOURCE_DIR}/Binomial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BinomialRng.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Exponential.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ExponentialRng.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Gamma.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GammaRng.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mt19937ar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Normal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/NormalRng.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Poisson.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PoissonRng.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Probability.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RandGenerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Uniform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/UniformRng.cpp
    )

