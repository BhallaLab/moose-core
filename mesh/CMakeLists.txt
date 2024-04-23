cmake_minimum_required(VERSION 3.20)
include( ${CMAKE_CURRENT_SOURCE_DIR}/../CheckCXXCompiler.cmake)
add_library(mesh
    ChemCompt.cpp
    MeshCompt.cpp
    MeshEntry.cpp
    CubeMesh.cpp
    CylBase.cpp
    CylMesh.cpp
    NeuroNode.cpp
    NeuroMesh.cpp
    SpineEntry.cpp
    SpineMesh.cpp
    PsdMesh.cpp
    EndoMesh.cpp
    PresynMesh.cpp
    testMesh.cpp
    )
