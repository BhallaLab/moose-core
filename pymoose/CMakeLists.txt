# Find Numpy 
add_definitions(-DPYMOOSE)
include_directories(../basecode ../msg)
include_directories(${NUMPY_INCLUDE_DIRS})
add_definitions(-DUSE_NUMPY)
add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

add_library(moosepython 
    vec.cpp
    mfield.cpp
    pymooseinit.cpp
    melement.cpp
    test_moosemodule.cpp 
    PyRun.cpp
    )
