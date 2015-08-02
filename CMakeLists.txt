set(CMAKE_LEGACY_CYGWIN_WIN32 0)
cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0004 OLD)

project(moose)
option(CMAKE_INSTALL_PREFIX "Install prefix" /usr)

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

## NOTE: PyMOOSE might break with c++11 support. Disable it.
#set(CMAKE_CXX_COMPILER "clang++")
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
    )

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P
    ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )

################################# OS SPECIFIC ##################################


################################# CMKAE MACROS #################################

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake_modules")
set(INSTALL_LIB_DIR lib CACHE PATH "${CMAKE_INSTALL_PREFIX}/lib")
set(INSTALL_BIN_DIR bin CACHE PATH "${CMAKE_INSTALL_PREFIX}/bin")
set(INSTALL_INCLUDE_DIR include CACHE PATH "${CMAKE_INSTALL_PREFIX}/include/")

########################### COMPILER MACROS #####################################

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

############################ BUILD CONFIGURATION #################################
set(VERIFY 1)
if(VERIFY)
    add_definitions(-DSANITY_CHECK -DARGS_CHECK -DRESULT_CHECK -DVALUE_CHECK)
endif(VERIFY)

# VERBOSITY OF OUTPUT
option(VERBOSITY "SET MOOSE VERBOSITY LEVEL" 0)
if(VERBOSITY)
    message("++ Verbosity of output is ${VERBOSITY}. This is not implemented
    yet!")
    add_definitions(-DVERBOSITY=${VERBOSITY})
else()
    message("++ Moose will be quiet")
    add_definitions(-DQUIET_MODE)
endif()

# Whether to notify the user after 1 second of simulation.
option(NOTIFY_PROGRESS "Show progress of simulation" ON)

# STATS INSIDE MOOSE
option(ENABLE_LOGGER  "Enable MOOSE logger" OFF)
if(ENABLE_LOGGER)
    message("++ LOGGER ENABLED")
    add_definitions(-DENABLE_LOGGER)
endif()

# Default macros
add_definitions(-DUSE_GENESIS_PARSER)

## Unit testing and debug mode.
option(DEBUG "Building with debugging support" OFF)
option(ENABLE_UNIT_TESTS "ENABLE UNIT TESTS IN MOOSE" ON)
if(ENABLE_UNIT_TESTS AND DEBUG)
    MESSAGE("++ Building for Debug/ Unit testing")
    add_definitions(-DDO_UNIT_TESTS)
    set(CMAKE_BUILD_TYPE Debug)
else()
    MESSAGE("++ Build for Relase/ No unit tests.")
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-UDO_UNIT_TESTS -O3)
endif()


#### DEPENDENCIES

find_package(GSL 1.16 REQUIRED)
option(ENABLE_SBML "Enable SBML support" ON)
if(ENABLE_SBML)
    find_package(LIBSBML)
    if(NOT LIBSBML_FOUND)
        MESSAGE("++ LIBSBML NOT FOUND.  Disabling SBML support.")
        ADD_DEFINITIONS(-UUSE_SBML)
    else()
        MESSAGE("++ SBML found. Support is ON: ${LIBSBML_LIBRARY}")
        ADD_DEFINITIONS(-DUSE_SBML)
        include_directories(${LIBSBML_INCLUDE_DIR})
        find_package(BZip2 REQUIRED)
        find_package(LibXML2 REQUIRED)
        find_package(ZLIB REQUIRED)
    endif()
else()
    SET(LIBSBML_FOUND FALSE)
ENDIF()

# python support: check for interpreter, sip, pyqt4
FIND_PACKAGE(PythonDev REQUIRED)

################################### SETUP BUILD ################################
include_directories(msg basecode)
add_executable(moose.bin basecode/main.cpp)
#add_library(moose SHARED basecode/main.cpp)
#set_target_properties(moose PROPERTIES COMPILE_FLAGS  "-DMOOSE_LIB")

## GSL
# Setup GSL
if(GSL_FOUND)
    add_definitions(-DUSE_GSL)
    include_directories(${GSL_INCLUDE_DIR})
endif()

## Setup hdf5
find_package(HDF5)
if(FOUND_HDF5)
    add_definitions(-DUSE_HDF5)
    include_directories(${HDF5_INCLUDE_DIR})
else(FOUND_HDF5)
    message("++ HDF5 not found. Disabling support")
endif(FOUND_HDF5)

find_package(Termcap)
find_package(Readline)

if(READLINE_FOUND AND TERMCAP_FOUND)
    add_definitions(-DUSE_READLINE)
    include_directories(${Readline_INCLUDE_DIR})
endif()

# Openmpi
find_package(MPI)
if(MPI_CXX_FOUND)
    message("++ Using MPI from ${MPI_CXX_INCLUDE_PATH}")
    include_directories(${MPI_CXX_INCLUDE_PATH})
    #set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    add_definitions(-DUSE_MPI)
else()
    message("++ Cound not find MPI")
    add_definitions(-UUSE_MPI)
endif()

############################ CYMOOSE related macros. ########################
if(BUILD_CYMOOSE)
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

# These are always shared libraries.
set(SYSTEM_LIBRARIES ${ZLIB_LIBRARIES} ${BZIP2_LIBRARIES} ${LibXML2_LIBRARIES} ${BZIP2_LIBRARIES})

# These libraries could be static of dynamic.
set(LIBRARIES "")

if(FOUND_HDF5)
    list(APPEND SYSTEM_LIBRARIES ${HDF5_LIBRARIES})
endif()

if(LIBSBML_FOUND)
    list(APPEND LIBRARIES moose_sbml ${LIBSBML_LIBRARY})
endif()

if(GSL_FOUND)
    list(APPEND LIBRARIES ${GSL_LIBRARIES})
endif()

if(READLINE_FOUND AND TERMCAP_FOUND)
    list(APPEND SYSTEM_LIBRARIES ${Readline_LIBRARY} ${TERMCAP_LIBRARY})
endif()

#if(MPI_CXX_FOUND)
#    list(APPEND SYSTEM_LIBRARIES ${MPI_CXX_LIBRARIES})
#endif()


###################################### LINKING #################################
set(MOOSE_LIBRARIES
    moose_builtins
    msg
    benchmarks
    shell
    randnum
    scheduling
    moose_mpi
    biophysics 
    utility 
    muparser
    kinetics 
    synapse
    intfire
    hsolve 
    mesh
    signeur
    tinyxml
    diffusion 
    ksolve
    device
    basecode
    )

target_link_libraries(moose.bin 
    "-Wl,--whole-archive"
    ${MOOSE_LIBRARIES}
    ${LIBRARIES}
    "-Wl,--no-whole-archive"
    ${SYSTEM_LIBRARIES}
    )

######################### BUILD PYMOOSE ########################################
option(BUILD_PYMOOSE "Build native python extension of MOOSE" 1)

# Root of all python module.
SET(PROJECT_PYTHON_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python)
if(BUILD_PYMOOSE)
    message("++ Building native python extension of MOOSE")
    set(PYMOOSE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/moose")

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
        LIBRARY_OUTPUT_DIRECTORY ${PYMOOSE_OUTPUT_DIRECTORY}
        PREFIX ""
        )
    list(APPEND LIBRARIES ${PYTHON_LIBRARIES})
    target_link_libraries(
        _moose 
        moosepython  
        "-Wl,--whole-archive"
        ${MOOSE_LIBRARIES}
        ${LIBRARIES}
        "-Wl,--no-whole-archive"
        ${SYSTEM_LIBRARIES}
        )

endif(BUILD_PYMOOSE)

######################### CYMOOSE #############################################
option(BUILD_CYMOOSE "Build python extension using Cython. Development version." 0)
if(BUILD_CYMOOSE)
    message("++ Building cython extension of MOOSE. Unstable and experimental.")
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
        ${LIBRARIES}
        "-Wl,--no-whole-archive"
        ${SYSTEM_LIBRARIES}
        )
endif()

######################################## MOOGLI ################################

# Moogli as external project
OPTION(BUILD_MOOGLI "Build moogli extension." OFF)
IF(BUILD_MOOGLI)
    SET(MOOGLI_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/moogli)

    FIND_PACKAGE(SIP REQUIRED)
    FIND_PACKAGE(PyQt4 REQUIRED)
    #FIND_PACKAGE(osgQt REQUIRED)
    MESSAGE("++ Building MOOGLI in ${MOOGLI_BUILD_PATH}")
    add_subdirectory(${PROJECT_SOURCE_DIR}/python/moogli)
    INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/python/moogli/include)
    LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/python/moogli)
else(BUILD_MOOGLI)
    MESSAGE("++ NOT building MOOGLI")
endif()

######################### INSTALL ##############################################

install(TARGETS moose.bin
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
    COMPONENT moose-gui
    )

# And the desktop file.
install(FILES ${PROJECT_SOURCE_DIR}/scripts/moose.desktop
    DESTINATION share/applications
    COMPONENT moose-gui
    )

# And the gui
install(DIRECTORY ${PROJECT_SOURCE_DIR}/gui
    DESTINATION lib/moose
    COMPONENT moose-gui
    )

install(FILES ${PROJECT_SOURCE_DIR}/python/setup-moose.py
    DESTINATION lib/moose
    COMPONENT moose-python
    )

## The demos goes here.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/Demos
    DESTINATION share/moose
    COMPONENT moose-gui
    )

# Both moose and moogli python directory are added here.
install(DIRECTORY ${PROJECT_SOURCE_DIR}/python/moose
    ${PROJECT_SOURCE_DIR}/python/libmumbl
    DESTINATION usr/share/moose
    COMPONENT moose-python
    )

install(FILES ${PROJECT_SOURCE_DIR}/python/setup-moose.py
    DESTINATION usr/share/moose
    COMPONENT moose-python
    )

if(BUILD_MOOGLI)

    SET(MOOGLI_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python/moogli)

    install(DIRECTORY ${PROJECT_SOURCE_DIR}/python/moogli
        DESTINATION usr/share/moogli
        COMPONENT moose-moogli
        )

    install(FILES ${PROJECT_SOURCE_DIR}/python/setup-moogli.py
        DESTINATION usr/share/moogli
        COMPONENT moose-moogli
        )

endif(BUILD_MOOGLI)

############################ CTEST ######################################

ENABLE_TESTING()

# If CTEST_OUTPUT_ON_FAILURE environment variable is set, the output is printed
# onto the console if a test fails.
SET(ENV{CTEST_OUTPUT_ON_FAILURE} ON)

if (CMAKE_CONFIGURATION_TYPES)
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} 
        --force-new-ctest-process --output-on-failure 
        --build-config "Debug")
else()
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} 
        --force-new-ctest-process
        --output-on-failure)
endif()

ADD_TEST(NAME moose.bin-raw-run
    COMMAND moose.bin -u -q
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

## PyMOOSE tests.

SET(TEST_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/python)

#ADD_TEST(NAME pymoose-test-mumbl
    #COMMAND ${PYTHON_EXECUTABLE} test_mumbl.py
    #WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/python
#    )

ADD_TEST(NAME pymoose-test-pymoose
    COMMAND ${PYTHON_EXECUTABLE} test_pymoose.py
    WORKING_DIRECTORY ${TEST_DIRECTORY}
    )

ADD_TEST(NAME pymoose-test-synchan
    COMMAND ${PYTHON_EXECUTABLE} test_synchan.py
    WORKING_DIRECTORY ${TEST_DIRECTORY}
    )

ADD_TEST(NAME pymoose-test-function
    COMMAND ${PYTHON_EXECUTABLE} test_function.py
    WORKING_DIRECTORY ${TEST_DIRECTORY}
    )

ADD_TEST(NAME pymoose-test-vec
    COMMAND ${PYTHON_EXECUTABLE} test_vec.py
    WORKING_DIRECTORY ${TEST_DIRECTORY}
    )

#ADD_TEST(NAME pymoose-test-sbml 
#    COMMAND ${PYTHON_EXECUTABLE} test_sbml.py
#    WORKING_DIRECTORY ${TEST_DIRECTORY}
#    )
#
#ADD_TEST(NAME pymoose-test-kkit 
#    COMMAND ${PYTHON_EXECUTABLE} test_kkit.py
#    WORKING_DIRECTORY ${TEST_DIRECTORY}
#    )
#

####### CPack ###############################

set(CPACK_GENERATOR "DEB;RPM;STGZ")
set(CPACK_STRIP_FILES TRUE)
set(CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE ON)
#  We need to compile python scripts on the installation  host.
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Dilawar Singh")
set(CPACK_PACKAGE_CONTACT "dilawars@ncbs.res.in")
#set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "MOOSE, The Neural Simulator")
set(CPACK_PACKAGE_VERSION "3.0.1")
set(CPACK_PACKAGE_VERSION_MAJOR "3")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "1")
set(CPACK_PACKAGE_VENDOR "NCBS Bangalore")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/README.md")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Dilawar Singh <dilawars@ncbs.res.in>")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "${PKGARCH}")
set(CPACK_PACKAGE_FILE_NAME "moose-${CPACK_PACKAGE_VERSION}-Linux-${PKGARCH}")
set(CPACK_STRIP_FILES moose)

set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_PACKAGE_DEPENDS 
    "openmpi-bin
    , python-matplotlib
    , python-qt4, openscenegraph
    , python-tornado, python-suds, python-nose, python-setuptools
    , libxml2"
)

set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    "${DEBIAN_POSTINST_PATH};${DEBIAN_PRERM_PATH};"
    )

## RPM 
set(CPACK_RPM_PACKAGE_ARCHITECTURE "${RPMPKGARCH}")
set(CPACK_RPM_SPEC_INSTALL_POST "/bin/true")
set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${RPM_POSTINST_PATH}")
set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE "${RPM_PRERM_PATH}")

# This has to be a single line.
set(CPACK_RPM_PACKAGE_REQUIRES
    "openmpi-devel python-matplotlib-qt4 python-qt4  python-tornado python-suds python-nose python-setuptools libxml2"
    )

# A workaround in RPM
set(CPACK_RPM_SPEC_MORE_DEFINE "%define ignore \#")
set(CPACK_RPM_USER_FILELIST "%ignore /" "%ignore /usr" "%ignore /usr/share" 
    "%ignore /usr/share/applications"
    "%ignore /tmp"
    )

include(CPack)

