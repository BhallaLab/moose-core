cmake_minimum_required(VERSION 2.6)
set(CMAKE_INSTALL_PREFIX /usr/local)

project(moose)

# This snippet is from LLVM project.

# Sanity check our source directory to make sure that we are not trying to
# generate an in-tree build (unless on MSVC_IDE, where it is ok), and to make
# sure that we don't have any stray generated files lying around in the tree
# (which would end up getting picked up by header search, instead of the correct
# versions).
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
    message(FATAL_ERROR 
        "In-source builds are not allowed.
        CMake would overwrite the makefiles distributed with Moose.
        Please create a directory and run cmake from there, passing the path
        to this source directory as the last argument.
        This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
        Please delete them."
        )
endif()

################################# CMKAE MACROS #################################

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
set(INSTALL_LIB_DIR lib CACHE PATH "/usr/local/lib")
set(INSTALL_BIN_DIR bin CACHE PATH "/usr/local/bin")
set(INSTALL_INCLUDE_DIR include CACHE PATH "/usr/local/include/")

########################### COMPILER MACROS #####################################
set(LOCAL_GSL 0)
if(LOCAL_GSL)
    add_subdirectory(gsl)
    include_directories(gsl)
else()
    find_package(GSL REQUIRED)
endif()

# Compiler 

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_COMPILER "clang++")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_COMPILER "g++")
else()
    message(WARNING "Continuing with unsupported compiler")
endif()

## Turn warning to error
add_definitions(-Wall
    #-Wno-return-type-c-linkage
    -Wno-unused-variable
    -Wno-unused-function
    #-Wno-unused-private-field
    )
add_definitions(-fPIC)

## Enable/Disable 2011 stupport.
set(ENABLE_STD_2011 0)
if(ENABLE_STD_2011)
    add_definitions(-DENABLE_STD_2011 -DDEBUG2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif(ENABLE_STD_2011)

# DO UNIT TESTS 
set(ENABLE_UNIT_TESTS 1)
if(ENABLE_UNIT_TESTS)
    MESSAGE("++ Unit testing enabled.")
    add_definitions(-DDO_UNIT_TESTS)
endif()

set(VERIFY 1)
if(VERIFY)
    add_definitions(-DSANITY_CHECK -DARGS_CHECK -DRESULT_CHECK -DVALUE_CHECK)
endif(VERIFY)

# VERBOSITY OF OUTPUT
set(VERBOSITY 0)
if(VERBOSITY)
    message("++ Verbosity of output is ${VERBOSITY}")
    add_definitions(-DVERBOSITY=${VERBOSITY})
else()
    #message("++ Moose will be quiet")
    #add_definitions(-DQUIET_MODE)
endif()

# STATS INSIDE MOOSE
set(ENABLE_LOGGER 1)
if(ENABLE_LOGGER)
    message("++ Moose will print statistics during simulation")
    add_definitions(-DENABLE_LOGGER)
endif()

# Default macros
add_definitions(-DUSE_GENESIS_PARSER)

set(DEBUG 0)
if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
else()
    set(CMAKE_BUILD_TYPE distribution)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -O3)
endif()

################################### SETUP BUILD ################################
include_directories(msg basecode)
add_executable(moose.bin basecode/main.cpp)

## GSL
add_definitions(-DUSE_GSL)
if(LOCAL_GSL)
    include_directories(gsl)
else()
    include_directories(${GSL_INCLUDE_DIR})
    target_link_libraries(moose.bin ${GSL_LIBRARIES})
endif()


## Setup hdf5
find_package(HDF5)
if(HDF5_FOUND)
    add_definitions(-DUSE_HDF5)
    include_directories(${HDF5_INCLUDE_DIR})
    target_link_libraries(moose.bin ${HDF5_LIBRARY})
endif(HDF5_FOUND)

find_package(BZip2)
find_package(LIBSBML)
if(LIBSBML_FOUND)
    # If LIBSBML_FOUND then we'll probably also need the LibXML2.
    add_definitions(-DUSE_SBML)
    find_package(LibXML2 REQUIRED)
    add_subdirectory(sbml)
    include_directories(${LIBSBML_INCLUDE_DIR})
    # Libxml2 is required.
    find_package(LibXML2 REQUIRED)
elseif()
    add_definitions(-UUSE_SBML)
endif(LIBSBML_FOUND)

find_package(Termcap)
find_package(Readline)
if(READLINE_FOUND AND TERMCAP_FOUND)
    add_definitions(-DUSE_READLINE)
    include_directories(${Readline_INCLUDE_DIR})
    target_link_libraries(moose.bin ${Readline_LIBRARY})
    target_link_libraries(moose.bin ${TERMCAP_LIBRARY})
endif()


# Add subdirectroeis
add_subdirectory(basecode)
add_subdirectory(biophysics)
add_subdirectory(msg)
add_subdirectory(shell)
add_subdirectory(randnum)
add_subdirectory(scheduling)
add_subdirectory(builtins)
add_subdirectory(utility)
add_subdirectory(external/muparser)
add_subdirectory(external/debug)
add_subdirectory(external/tinyxml)
add_subdirectory(mesh)
add_subdirectory(mpi)
add_subdirectory(signeur)
add_subdirectory(ksolve)
add_subdirectory(hsolve)
add_subdirectory(diffusion)
add_subdirectory(device)
add_subdirectory(benchmarks)
add_subdirectory(kinetics)

###################################### LINKING #################################
target_link_libraries(moose.bin 
    "-Wl,--whole-archive"
    moose_builtins
    basecode
    msg
    shell
    randnum
    scheduling
    moose_mpi
    utility 
    muparser
    biophysics 
    kinetics 
    ksolve
    hsolve 
    mesh
    signeur
    tinyxml
    diffusion 
    device
    benchmarks
    "-Wl,--no-whole-archive"
    )

if(LIBSBML_FOUND)
    target_link_libraries(moose.bin moose_sbml ${LIBSBML_LIBRARY} z)
    if(BZIP2_FOUND)
        target_link_libraries(moose.bin ${BZIP2_LIBRARIES})
    else()
        target_link_libraries(moose.bin bz2)
    endif()
    if(LIBXML2_FOUND)
        target_link_libraries(moose.bin ${LibXML2_LIBRARIES})
    endif()
endif()

# Link gsl 
set(LOCAL_GSL_LIBRARIES  gsl gslcblas m)
target_link_libraries(moose.bin ${LOCAL_GSL_LIBRARIES})

######################### BUILD PYMOOSE ########################################
set(BUILD_PYMOOSE 1)
if(BUILD_PYMOOSE)
    find_package(NumpyHeaders)
    if(PYTHON_NUMPY_FOUND)
        add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
        include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
        add_definitions(-DUSE_NUMPY)
    else()
        message(WARNING "Could not find numpy/arrayobject.h in usual places")
        add_definitions(-UUSE_NUMPY)
    endif()
    # Openmpi
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})

    find_package(PythonDev REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    add_subdirectory(pymoose)
    add_library(_moose SHARED pymoose/moosemodule.cpp)
    set_target_properties(_moose PROPERTIES COMPILE_DEFINITIONS "PYMOOSE")
    set_target_properties(_moose PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/moose"
        PREFIX ""
        )

    set(LIBRARIES ${PYTHON_LIBRARY} ${LibBZip2_LIBRARIES} 
        ${LOCAL_GSL_LIBRARIES}
        )
    if(HDF5_FOUND)
        list(APPEND LIBRARIES ${HDF5_LIBRARY})
    endif()

    if(LIBSBML_FOUND)
        list(APPEND LIBRARIES moose_sbml ${LIBSBML_LIBRARY})
    endif()

    if(LIBXML2_FOUND)
        list(APPEND LIBSBML_FOUND ${LibXML2_LIBRARIES})
    endif()

    target_link_libraries(_moose 
        "-Wl,--whole-archive"
        moosepython
        basecode
        msg
        shell
        randnum
        scheduling
        moose_mpi
        moose_builtins
        utility 
        muparser
        biophysics 
        kinetics 
        ksolve
        tinyxml
        hsolve 
        mesh
        signeur
        diffusion 
        device
        benchmarks
        pthread
        ${LIBRARIES}
        "-Wl,--no-whole-archive"
        )
endif(BUILD_PYMOOSE)

######################### INSTALL ##############################################
install(TARGETS moose.bin
    RUNTIME DESTINATION bin
    )
install(PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/moose 
    DESTINATION bin
    )

######################  TESTING  ###############################################
include(CTest)
find_package(PythonInterp REQUIRED)
set(ENV{PYTHONPATH} "${CMAKE_SOURCE_DIR}/python")
message("++ PYTHONPATH is set to $ENV{PYTHONPATH}")

add_custom_target(check_moose
    COMMAND ${CMAKE_BINARY_DIR}/run_moose.sh --output-on-failure
    COMMENT "Checking: moose with test-enabled" VERBATIM 
    )

add_custom_target(check_python
    #COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_SOURCE_DIR}/python/moose/test.py"
    #COMMENT "Checking: loading moose into python" VERBATIM
    COMMAND ${PYTHON_EXECUTABLE} test_pymoose.py 
    COMMENT "Checking: running test_pymoose in python" VERBATIM
    COMMAND ${PYTHON_EXECUTABLE} Rallpacks/rallpacks_passive_cable.py
    COMMENT "Checking: running rallpack2"
    #COMMAND ${PYTHON_EXECUTABLE} Rallpacks/rallpacks_passive_cable.py
    #COMMENT "Checking: running rallpack2"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/python"
    )


