cmake_minimum_required(VERSION 2.8)
set(CMAKE_INSTALL_PREFIX /usr)
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

set(CMAKE_VERBOSE_MAKEFILE FALSE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
set(INSTALL_LIB_DIR lib CACHE PATH "${CMAKE_INSTALL_PREFIX}/lib")
set(INSTALL_BIN_DIR bin CACHE PATH "${CMAKE_INSTALL_PREFIX}/bin")
set(INSTALL_INCLUDE_DIR include CACHE PATH "${CMAKE_INSTALL_PREFIX}/include/")

########################### COMPILER MACROS #####################################
# Compiler 

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_COMPILER "clang++")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_COMPILER "g++")
else()
    message(WARNING "Continuing with unsupported compiler: ${CMAKE_CXX_COMPILER}")
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
set(ENABLE_UNIT_TESTS 0)
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
set(ENABLE_LOGGER 0)
if(ENABLE_LOGGER)
    message("++ LOGGER ENABLED")
    add_definitions(-DENABLE_LOGGER)
endif()

# Default macros
add_definitions(-DUSE_GENESIS_PARSER)

set(DEBUG 0)
if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
else()
    set(CMAKE_BUILD_TYPE distribution)
    add_definitions(-UDO_UNIT_TESTS)
endif()

# Wheather to use local-gsl or not. link_directories are need to be called
# before target is created therefore it is here.
if(EXISTS ${CMAKE_SOURCE_DIR}/external/gsl-1.16/gsl-config.in)
    MESSAGE("++ Local gsl is found.")
    set(LOCAL_GSL 1)
endif()

if(LOCAL_GSL)
    set(GSL_SOURCE_DIR ${CMAKE_SOURCE_DIR}/external/gsl-1.16)
    set(GSL_INSTALL_DIR ${CMAKE_SOURCE_DIR}/gsl)
    include(ExternalProject)
    ExternalProject_Add(GSL
        PREFIX ${GSL_SOURCE_DIR}
        SOURCE_DIR ${GSL_SOURCE_DIR}
        CMAKE_ARGS ${CMARGS} -DCMAKE_INSTALL_PREFIX=${GSL_INSTALL_DIR}
        )
    include_directories(${GSL_INSTALL_DIR}/include)
    LINK_DIRECTORIES(${GSL_INSTALL_DIR}/lib)
    set(LOCAL_GSL_LIBRARIES gsl gslcblas)
else()
    find_package(GSL 1.16 REQUIRED)
endif()

################################### SETUP BUILD ################################
include_directories(msg basecode)
add_executable(moose.bin basecode/main.cpp)

## GSL
# Setup GSL
if(LOCAL_GSL)
    set(GSL_FOUND 1)
    add_definitions(-DUSE_GSL)
    add_dependencies(moose.bin GSL)
else()
    if(GSL_FOUND)
        include_directories(${GSL_INCLUDE_DIR})
        add_definitions(-DUSE_GSL)
    endif()
endif()


## Setup hdf5
find_package(HDF5)
if(HDF5_FOUND)
    add_definitions(-DUSE_HDF5)
    include_directories(${HDF5_INCLUDE_DIR})
endif(HDF5_FOUND)

find_package(LIBSBML)
if(LIBSBML_FOUND)
    # If LIBSBML_FOUND then we'll probably also need the LibXML2.
    add_definitions(-DUSE_SBML)
    find_package(LibXML2 REQUIRED)
    add_subdirectory(sbml)
    include_directories(${LIBSBML_INCLUDE_DIR})
elseif()
    add_definitions(-UUSE_SBML)
endif(LIBSBML_FOUND)

find_package(Termcap)
find_package(Readline)

if(READLINE_FOUND AND TERMCAP_FOUND)
    add_definitions(-DUSE_READLINE)
    include_directories(${Readline_INCLUDE_DIR})
endif()

# Openmpi
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})

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

set(LIBRARIES ${LibBZip2_LIBRARIES})
if(HDF5_FOUND)
    list(APPEND LIBRARIES ${HDF5_LIBRARY})
endif()

if(LIBSBML_FOUND)
    list(APPEND LIBRARIES moose_sbml ${LIBSBML_LIBRARY})
endif()

if(LIBXML2_FOUND)
    list(APPEND LIBRARIES ${LibXML2_LIBRARIES})
endif()

if(GSL_FOUND)
    # Link gsl 
    list(APPEND LIBRARIES ${GSL_LIBRARIES})
endif()

if(READLINE_FOUND AND TERMCAP_FOUND)
    list(APPEND LIBRARIES ${Readline_LIBRARY} ${TERMCAP_LIBRARY})
endif()

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
    hsolve 
    mesh
    signeur
    tinyxml
    diffusion 
    device
    benchmarks
    ksolve
    "-Wl,--no-whole-archive"
    ${LOCAL_GSL_LIBRARIES}
    ${LIBRARIES}
    )

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

    find_package(PythonDev REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    add_subdirectory(pymoose)
    add_library(_moose SHARED pymoose/moosemodule.cpp)

    if(LOCAL_GSL)
        add_dependencies(_moose GSL)
    endif()

    set_target_properties(_moose PROPERTIES COMPILE_DEFINITIONS "PYMOOSE")
    set_target_properties(_moose PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/moose"
        PREFIX ""
        )
    list(APPEND LIBRARIES ${PYTHON_LIBRARIES})

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
        "-Wl,--no-whole-archive"
        ${LOCAL_GSL_LIBRARIES}
        ${LIBRARIES}
        )
endif(BUILD_PYMOOSE)

######################### INSTALL ##############################################
install(TARGETS moose.bin 
    RUNTIME DESTINATION bin
    COMPONENT applications
    )
install(PROGRAMS scripts/moose DESTINATION bin
    COMPONENT applications
    )
install(DIRECTORY gsl/lib/ DESTINATION lib
    COMPONENT libraries
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING PATTERN "*.so*"
    PATTERN ".svn" EXCLUDE
    )
install(SCRIPT python/InstallPyMoose.cmake
    COMPONENT pymoose
    )
install(PROGRAMS scripts/moosegui 
    DESTINATION bin
    )
########################## PACKAGING #########################################

set(CPACK_GENERATOR "DEB")
set(CPACK_STRIP_FILES TRUE)

#  We need to compile python scripts on the installation  host.
set(CPACK_COMPONENTS_ALL applications libraries)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Dilawar Singh")
set(CPACK_PACKAGE_CONTACT "dilawars@ncbs.res.in")
#set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Moose")
set(CPACK_INSTALL_SCRIPT "${CMAKE_SOURCE_DIR}/python/InstallPyMoose.cmake")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VENDOR "NCBS Bangalore")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Dilawar Singh <dilawars@ncbs.res.in>")
if(WIN32)
    MESSAGE("++ Currently only Unix is supported")
else()
    set(CPACK_STRIP_FILES moose.bin)
endif()
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_PACKAGE_FILE_NAME
    "moose_${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.orig"
    )
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libgsl (>= 1.16) libopenmpi libhdf5-7 python-numpy")
include(CPack)

######################  TESTING  ###############################################
include(CTest)
find_package(PythonInterp REQUIRED)
set(ENV{PYTHONPATH} "${CMAKE_SOURCE_DIR}/python")
message("++ PYTHONPATH is set to $ENV{PYTHONPATH}")

add_test(NAME check_moose
    WORKING_DIRECTORY .
    COMMAND ./run_moose.sh
    )
