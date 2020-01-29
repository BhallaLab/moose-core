add_definitions(-DPYMOOSE)
include_directories(../basecode ../msg)

set(PYMOOSE_SRCS
    moosemodule.cpp
    vec.cpp
    mfield.cpp
    pymooseinit.cpp
    melement.cpp
    test_moosemodule.cpp
    PyRun.cpp
    )

if(WITH_PYTHON2)
    include_directories(${Python_NumPy_INCLUDE_DIRS} ${Python2_INCLUDE_DIRS})
    execute_process(
        COMMAND ${Python2_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE Python_NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
elseif(WITH_PYTHON)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE Python_NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    include_directories(${Python_NumPy_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})
endif()

message(STATUS "NumPy include path: ${Python_NumPy_INCLUDE_DIRS}")

add_library( _moose MODULE ${PYMOOSE_SRCS} )
set(PYMOOSE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/moose")
add_definitions(-DUSE_NUMPY)
add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
target_include_directories(_moose PRIVATE ${Python_INCLUDE_DIRS})

set_target_properties(_moose PROPERTIES 
    COMPILE_DEFINITIONS "PYMOOSE"
    LIBRARY_OUTPUT_DIRECTORY ${PYMOOSE_OUTPUT_DIRECTORY}
    PREFIX ""
    SUFFIX ${PYTHON_SO_EXTENSION}
    )

if(NOT(PYTHON_SO_EXTENSION STREQUAL ""))
    set_target_properties(_moose PROPERTIES
    SUFFIX ${PYTHON_SO_EXTENSION})
endif()

# see issue #80
if(HDF5_LIBRARY_DIRS)
    set_target_properties( _moose PROPERTIES LINK_FLAGS "-L${HDF5_LIBRARY_DIRS}" )
endif()


if(MACOSX)
    set(CMAKE_MODULE_LINKER_FLAGS "-undefined dynamic_lookup")
    message(STATUS "ADDING some linker flags ${CMAKE_EXE_LINKER_FLAGS}")
endif(MACOSX)

if(MACOSX)
    target_link_libraries( _moose
        "-Wl,-all_load"
        ${MOOSE_LIBRARIES}
        ${STATIC_LIBRARIES}
        )
    target_link_libraries(_moose
        ${SYSTEM_SHARED_LIBS}
        )
else(MACOSX)
    target_link_libraries(_moose
        "-Wl,--whole-archive"
        ${MOOSE_LIBRARIES}
        ${STATIC_LIBRARIES}
        "-Wl,--no-whole-archive"
        # ${PYTHON_LIBRARIES}
        ${SYSTEM_SHARED_LIBS}
        )
endif(MACOSX)
add_custom_command(TARGET _moose POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan
        "MOOSE python extention is successfully built. Now "
        " Run 'sudo make install' to install it. "
        " "
        "NOTE: Run 'pip uninstall moose' to uninstall moose."
    VERBATIM
    )

