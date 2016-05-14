add_definitions(-DPYMOOSE)

include_directories(../basecode ../msg)

execute_process( COMMAND ${PYTHON_EXECUTABLE}-config --includes
    OUTPUT_VARIABLE PYTHON_INCLUDE_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
execute_process( COMMAND ${PYTHON_EXECUTABLE}-config --libs
    OUTPUT_VARIABLE PYTHON_LIBRARIES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
message( STATUS "Python include flags: ${PYTHON_INCLUDE_FLAGS}" )

add_library(moosepython 
    vec.cpp
    mfield.cpp
    pymooseinit.cpp
    melement.cpp
    test_moosemodule.cpp 
    PyRun.cpp
    )

set_target_properties( moosepython 
    PROPERTIES COMPILE_FLAGS "${PYTHON_INCLUDE_FLAGS}"
    )
