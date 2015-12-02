# Find Numpy 
add_definitions(-DPYMOOSE)
include_directories(../basecode ../msg)
include_directories(${NUMPY_INCLUDE_DIRS})
add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
add_definitions(-DUSE_NUMPY)

add_library(moosepython 
    vec.cpp
    mfield.cpp
    pymooseinit.cpp
    melement.cpp
    test_moosemodule.cpp 
    PyRun.cpp
    )

#target_link_libraries(moosepython utility basecode biophysics)
#if(USE_SBML)
#    target_link_libraries(moosepython moose_sbml)
#endif(USE_SBML)
#if(USE_HDF5)
#    target_link_libraries(moosepython hdf5)
#endif(USE_HDF5)
