# Find Numpy 
include_directories(../basecode ../msg ../external/debug external/debug/)
add_definitions(-DPYMOOSE)
add_library(moosepython 
    vec.cpp
    mfield.cpp
    pymooseinit.cpp
    melement.cpp
    test_moosemodule.cpp 
    )
target_link_libraries(moosepython utility basecode biophysics)

if(USE_SBML)
    target_link_libraries(moosepython moose_sbml)
endif(USE_SBML)
if(USE_HDF5)
    target_link_libraries(moosepython hdf5)
endif(USE_HDF5)
