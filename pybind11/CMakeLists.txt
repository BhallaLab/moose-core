# Add pybind11 module.

execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version
    OUTPUT_VARIABLE COMPILER_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE)

pybind11_add_module(_moose 
    pymoose.cpp 
    helper.cpp 
    Finfo.cpp
    MooseVec.cpp 
    PyRun.cpp)

add_dependencies(_moose libmoose)

# Use in version_info dict.
target_compile_definitions(_moose PRIVATE
    COMPILER_STRING="${CMAKE_CXX_COMPILER_ID},${CMAKE_CXX_COMPILER},${CMAKE_CXX_COMPILER_VERSION}")

set_target_properties(_moose PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/moose
    PREFIX ""
    SUFFIX ".so")

if(APPLE)
    set(CMAKE_MODULE_LINKER_FLAGS "-undefined dynamic_lookup")
    message(STATUS "ADDING some linker flags ${CMAKE_EXE_LINKER_FLAGS}")
    # cmake --help-policy CMP0042
    set_target_properties(_moose PROPERTIES MACOSX_RPATH OFF)
endif(APPLE)

if(APPLE)
    # OSX
    target_link_libraries(_moose PRIVATE
        "-Wl,-all_load"
        ${MOOSE_LIBRARIES}
        ${STATIC_LIBRARIES})
    target_link_libraries(_moose PRIVATE ${SYSTEM_SHARED_LIBS})
else(APPLE)
    # Linux
    target_link_libraries(_moose PRIVATE
        "-Wl,--whole-archive" 
        ${MOOSE_LIBRARIES} 
        ${STATIC_LIBRARIES} 
        "-Wl,--no-whole-archive"
        ${SYSTEM_SHARED_LIBS})
endif()


# Copy python tree to BUILD directory. User can set PYTHONPATH to
# ${CMAKE_BINARY_DIR}/python. 
set(PYMOOSE_BUILD_DIR ${CMAKE_BINARY_DIR}/python)
add_custom_target(copy_python_tree ALL
    # Copy whole Python's source tree to build directory.
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/python/
        ${PYMOOSE_BUILD_DIR}
    COMMENT "pybind11: Copying Python src ${CMAKE_SOURCE_DIR}/python → ${PYMOOSE_BUILD_DIR}"
    DEPENDS _moose 
    VERBATIM)

 
# Create a binary distribution inside a directory. Installation will copy that
# directory.
set(_platform "CMAKE")
set(PYMOOSE_BDIST_FILE ${CMAKE_BINARY_DIR}/pymoose-${VERSION_MOOSE}.${_platform}.tar.gz)
set(PYMOOSE_INSTALL_DIR ${CMAKE_BINARY_DIR}/_pymoose_temp_install)
file(MAKE_DIRECTORY ${PYMOOSE_INSTALL_DIR})

add_custom_command(OUTPUT ${PYMOOSE_BDIST_FILE}
  COMMAND ${PYTHON_EXECUTABLE} setup.py build_py  
  COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_dumb 
    --skip-build -p "${_platform}" -d ${CMAKE_BINARY_DIR} 
  COMMAND ${CMAKE_COMMAND} -E chdir ${PYMOOSE_INSTALL_DIR} tar xf ${PYMOOSE_BDIST_FILE}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "python's binary distribution is saved to ${CMAKE_BINARY_DIR}"
  VERBATIM)

add_custom_target(pymoose_sdist ALL
    DEPENDS ${PYMOOSE_BDIST_FILE} _moose
    COMMENT "Building pymoose source distribution.")

install(DIRECTORY ${PYMOOSE_INSTALL_DIR}/usr/local/
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    CONFIGURATIONS Debug Release)
