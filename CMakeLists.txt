cmake_minimum_required(VERSION 2.6)

project(moose)

SET(CMAKE_INSTALL_PREFIX /usr)

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

set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")
set(INSTALL_LIB_DIR lib CACHE PATH "${CMAKE_INSTALL_PREFIX}/lib")
set(INSTALL_BIN_DIR bin CACHE PATH "${CMAKE_INSTALL_PREFIX}/bin")
set(INSTALL_INCLUDE_DIR include CACHE PATH "${CMAKE_INSTALL_PREFIX}/include/")

########################### COMPILER MACROS #####################################
# Compiler 

#if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    #set(CMAKE_CXX_COMPILER "clang++")
#elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    #set(CMAKE_CXX_COMPILER "g++")
#else()
    #message(WARNING "Continuing with unsupported compiler: ${CMAKE_CXX_COMPILER}")
#endif()

## Turn warning to error
add_definitions(-Wall
    #-Wno-return-type-c-linkage
    -Wno-unused-variable
    -Wno-unused-function
    #-Wno-unused-private-field
    )
add_definitions(-fPIC)

## Enable/Disable 2011 stupport.
include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

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
option(ENABLE_LOGGER  "Enable MOOSE logger" ON)
if(ENABLE_LOGGER)
    message("++ LOGGER ENABLED")
    add_definitions(-DENABLE_LOGGER)
endif()

# Default macros
add_definitions(-DUSE_GENESIS_PARSER)

option(DEBUG "Building with debugging support" 0)
if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
else()
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-UDO_UNIT_TESTS -O2)
endif()

find_package(GSL 1.16 REQUIRED)

find_package(LIBSBML)
if(NOT LIBSBML_FOUND)
    MESSAGE("+ LIBSBML NOT FOUND.  Disabling SBML support.")
    ADD_DEFINITIONS(-UUSE_SBML)
else()
    ADD_DEFINITIONS(-DUSE_SBML)
    include_directories(${LIBSBML_INCLUDE_DIR})
    find_package(BZip2 REQUIRED)
    find_package(LibXML2 REQUIRED)
endif()

# python support: check for interpreter, sip, pyqt4
FIND_PACKAGE(PythonDev REQUIRED)
FIND_PACKAGE(SIP REQUIRED)
FIND_PACKAGE(PyQt4 REQUIRED)
FIND_PACKAGE(osgQt REQUIRED)

################################### SETUP BUILD ################################
include_directories(msg basecode)
add_executable(moose-bin basecode/main.cpp)
add_library(moose SHARED basecode/main.cpp)
set_target_properties(moose PROPERTIES COMPILE_FLAGS  "-DMOOSE_LIB")

## GSL
# Setup GSL
if(GSL_FOUND)
    add_definitions(-DUSE_GSL)
    add_dependencies(moose GSL)
endif()


## Setup hdf5
find_package(HDF5)
if(HDF5_FOUND)
    add_definitions(-DUSE_HDF5)
    include_directories(${HDF5_INCLUDE_DIR})
endif(HDF5_FOUND)

find_package(Termcap)
find_package(Readline)

if(READLINE_FOUND AND TERMCAP_FOUND)
    add_definitions(-DUSE_READLINE)
    include_directories(${Readline_INCLUDE_DIR})
endif()

# Openmpi
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})

############################ CYMOOSE related macros. ########################
set(CYMOOSE 1)
if(CYMOOSE)
    add_definitions("-DCYMOOSE -DCYTHON")
endif()

# Add subdirectroeis
add_subdirectory(basecode)
add_subdirectory(msg)
add_subdirectory(shell)
add_subdirectory(randnum)
add_subdirectory(scheduling)
add_subdirectory(biophysics)
add_subdirectory(builtins)
add_subdirectory(utility)
add_subdirectory(external/muparser)
add_subdirectory(external/debug)
add_subdirectory(external/tinyxml)
add_subdirectory(mesh)
add_subdirectory(sbml)
add_subdirectory(mpi)
add_subdirectory(signeur)
add_subdirectory(ksolve)
add_subdirectory(hsolve)
add_subdirectory(diffusion)
add_subdirectory(device)
add_subdirectory(benchmarks)
add_subdirectory(kinetics)
add_subdirectory(synapse)
add_subdirectory(intfire)
add_subdirectory(examples)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/moogli/include)
LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/moogli/lib)



set(LIBRARIES ${BZIP2_LIBRARIES} ${LibXML2_LIBRARIES})
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
    list(APPEND LIBRARIES ${GSL_LIBRARIES})
endif()

if(READLINE_FOUND AND TERMCAP_FOUND)
    list(APPEND LIBRARIES ${Readline_LIBRARY} ${TERMCAP_LIBRARY})
endif()

###################################### LINKING #################################
set(MOOSE_LIBRARIES
    moose_builtins
    msg
    benchmarks
    shell
    randnum
    scheduling
    moose_mpi
    utility 
    muparser
    kinetics 
    hsolve 
    mesh
    signeur
    tinyxml
    biophysics 
    synapse
    diffusion 
    basecode
    device
    ksolve
    intfire
    )

target_link_libraries(moose
    ${MOOSE_LIBRARIES}
    ${LOCAL_GSL_LIBRARIES}
    ${LIBRARIES}
    )

target_link_libraries(moose-bin 
    moose
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
    target_link_libraries(
        _moose 
        moosepython  
        "-Wl,--whole-archive"
        ${MOOSE_LIBRARIES}
        "-Wl,--no-whole-archive"
        ${LIBRARIES}
        )

endif(BUILD_PYMOOSE)

######################### CYMOOSE #############################################
if(CYMOOSE)
    add_library(cymoose SHARED cymoose/cymoose_methods.cxx)
    set_target_properties(moose PROPERTIES COMPILE_DEFINITIONS "CYMOOSE")
    if(DEBUG)
        add_definitions("-g -Wall")
    endif()

    set_target_properties(cymoose PROPERTIES 
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/cymoose"
            )

    target_link_libraries(cymoose 
        "-Wl,--whole-archive"
        ${MOOSE_LIBRARIES}
        "-Wl,--no-whole-archive"
        ${LIBRARIES}
        )
endif()

######################################## MOOGLI ################################
# Moogli as external project
OPTION(BUILD_MOOGLI "Build moogli extension." ON)
IF(BUILD_MOOGLI)
    MESSAGE("++ Building MOOGLI")
    #SET(MOOGL_SRC_PATH ${PROJECT_SOURCE_DIR}/python/moogli)
    SET(MOOGLI_BUILD_PATH ${CMAKE_BINARY_DIR}/_moogli)
    #FILE(MAKE_DIRECTORY ${MOOGLI_BUILD_PATH})
    #ADD_CUSTOM_TARGET(moogli ALL
        #COMMAND ${CMAKE_COMMAND} ${MOOGL_SRC_PATH}
        #COMMAND make
        #WORKING_DIRECTORY ${MOOGLI_BUILD_PATH}
        #COMMENT "Building and installing MOOGLI"
        #VERBATIM
    #) 
    add_subdirectory(${PROJECT_SOURCE_DIR}/python/moogli)
    INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/python/moogli/include)
    LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/python/moogli)
else()
    MESSAGE("++ NOT building MOOGLI")
endif()

######################### INSTALL ##############################################
install(TARGETS moose
    LIBRARY DESTINATION lib
    COMPONENT moose-core
    )

install(TARGETS moose-bin
    DESTINATION bin
    COMPONENT moose-core
    )

install(PROGRAMS ${PROJECT_SOURCE_DIR}/scripts/moose
    DESTINATION bin
    COMPONENT moose-core
    )

install(FILES ${PROJECT_SOURCE_DIR}/scripts/moose
    DESTINATION bin
    COMPONENT moose-core
    )

install(PROGRAMS ${PROJECT_SOURCE_DIR}/scripts/moosegui
    DESTINATION bin
    COMPONENT moose-gui
    )

## Install the icon
install(FILES ${PROJECT_SOURCE_DIR}/gui/icons/moose_icon.png 
    DESTINATION share/moose
    COMPONENT moose-python
    )

# And the desktop file.
install(FILES ${PROJECT_SOURCE_DIR}/scripts/moose.desktop
    DESTINATION share/applications
    COMPONENT moose-python
    )

# And the gui
install(DIRECTORY ${PROJECT_SOURCE_DIR}/gui
    DESTINATION lib/moose
    COMPONENT moose-gui
    )

# and the python dependencies
install(DIRECTORY ${PROJECT_SOURCE_DIR}/python/moose
    DESTINATION lib/moose/python
    COMPONENT moose-python
    )

# and the python dependencies
install(DIRECTORY ${PROJECT_SOURCE_DIR}/python/libmumbl
    DESTINATION lib/moose/python
    COMPONENT moose-python
    )

INSTALL(FILES ${PROJECT_SOURCE_DIR}/python/setup.py
    DESTINATION lib/moose
    COMPONENT moose-python
    )

## The demos goes here.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Demos
    DESTINATION share/moose
    COMPONENT moose-python
    )


if(BUILD_MOOGLI)

    SET(MOOGLI_TEMP_DIR ${MOOGLI_BUILD_PATH})

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/python/moogli/_moogli/moogli.so
        DESTINATION ${MOOGLI_TEMP_DIR}/moogli
        COMPONENT moose
        )

    install(FILES ${PROJECT_SOURCE_DIR}/python/moogli/__init__.py
        DESTINATION ${MOOGLI_TEMP_DIR}/moogli
        COMPONENT moose
        )

    install(FILES ${PROJECT_SOURCE_DIR}/python/setup-moogli.py
        DESTINATION ${MOOGLI_TEMP_DIR}
        COMPONENT moose
        )

    install(CODE
        "execute_process(COMMAND ${PYTHON_EXECUTABLE} setup-moogli.py install 
        WORKING_DIRECTORY ${MOOGLI_TEMP_DIR})"
        COMPONENT moose
        )
    
    install(CODE "MESSAGE(\"++ Installed moogli using setuptools. You can
        uninstall it using python-pip\")"
        )

endif(BUILD_MOOGLI)
