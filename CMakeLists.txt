set(CMAKE_LEGACY_CYGWIN_WIN32 0)
cmake_minimum_required(VERSION 2.6.0 FATAL_ERROR)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    cmake_policy(SET CMP0004 OLD)
    if(POLICY CMP0050)
        cmake_policy(SET CMP0050 OLD)
    endif(POLICY CMP0050)
endif(COMMAND cmake_policy)

project(moose)

# Set moose version
# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git describe --tags --long
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_VERSION_OUTPUT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions( -DMOOSE_VERSION="${GIT_VERSION_OUTPUT}")

include(FindPkgConfig)

# This snippet is from LLVM project.
# Sanity check our source directory to make sure that we are not trying to
# generate an in-tree build (unless on MSVC_IDE, where it is ok), and to make
# sure that we don't have any stray generated files lying around in the tree
# (which would end up getting picked up by header search, instead of the correct
# versions).

message(STATUS "CMAKE_INSTALL_PREFIX : ${CMAKE_INSTALL_PREFIX}")
if( CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
    message(FATAL_ERROR 
        "======================================================================\n"
        "In-source builds are not allowed. Remove CMakeCache.txt and CMakeFiles\n"
        "directory and do something like this inside this directory \n"
        "    $ mkdir _build_dir \n"
        "    $ cd _build_dir  \n"
        "    $ cmake ..  \n"
        "===================================================================== \n"
        )
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

# This is for testing purpose.
link_directories(${CMAKE_CURRENT_BINARY_DIR})

################################# OS Specific ##################################
message(STATUS "Operating system: ${CMAKE_SYSTEM_NAME}")
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    CMAKE_POLICY(SET CMP0042 NEW)
    set(MACOSX TRUE)
else()
    set(MACOSX FALSE)
endif()

################################# CMAKE OPTIONS ##################################

option(WITH_DOC "Build documentation using python-sphinx and doxygen" OFF)

## Unit testing and debug mode.
option(DEBUG "Build with debug support" ON)
option(GPROF "Build for profiling using gprof" OFF)
option(ENABLE_UNIT_TESTS "Enable unit tests (DEBUG should also be ON)" ON)
option(WITH_PYTHON "Build native python extension" ON)
option(WITH_MPI  "Enable Openmpi support" OFF)
option(WITH_CURSES "Enable ncurses support (deprecated)" OFF)
option(WARN_AS_ERROR "Treat warnings as error while compiling" OFF)
option(USE_BOOST "Use boost library (experimental, development)" ON)

# If SBML is available, it will automaticall enable the support. If ON, then
# libSBML must be present.
option(WITH_SBML  "Enable SBML support. Automatically detected." OFF)

option(SBML_STATIC_HOME 
    "Path prefix where static sbml library can be found e.g. /opt/sbml/"
    OFF
    )
option(HDF5_STATIC_HOME
    "Path prefix where static hdf5 library can be found e.g /opt/sw/hdf5 "
    OFF
    )

################################# CMKAE MACROS #################################

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

########################### COMPILER MACROS #####################################

# Turn warning to error: Not all of the options may be supported on all
# versions of compilers. be careful here.
add_definitions(-fPIC)

if(WARN_AS_ERROR)
    message(STATUS "Treating compiler warnings as errors" )
    add_definitions( -Wall
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-macro-redefined
        -Wno-unused-local-typedefs
        -Werror 
        )
else(WARN_AS_ERROR)
    add_definitions(-Wall
        #-Wno-return-type-c-linkage
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-unused-local-typedefs
        #-Wno-unused-private-field
        )
endif(WARN_AS_ERROR)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    message(STATUS "Your compiler supports c++11 features. Enabling it")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    message(STATUS "Your compiler supports c++0x features. Enabling it")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()


############################ BUILD CONFIGURATION #################################

# Default macros
add_definitions(-DUSE_GENESIS_PARSER)

if(ENABLE_UNIT_TESTS)
    message(STATUS "Building for Debug/Unit testing")
    add_definitions(-DDO_UNIT_TESTS -DVERBOSITY=1)
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-g)
elseif(DEBUG)
    MESSAGE(STATUS "Enabled Unit tests")
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DDO_UNIT_TESTS -DVERBOSITY=1)
else()
    message(STATUS "Building for Release/No unit tests.")
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-UDO_UNIT_TESTS -O3)
endif()

if(GPROF AND DEBUG)
    message(STATUS "Compiling with profiling with gprof")
    add_definitions(-pg)
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-pg")
endif()

IF(NOTIFY_PROGRESS)
    IF(COMPILER_SUPPORTS_CXX11 OR COMPILER_SUPPORTS_CXX0X)
        MESSAGE(STATUS "MOOSE will occasionally write to __moose__progress__ file "
            " status of simulation (done time/total time)"
            )
    ELSE()
        MESSAGE(FATAL_ERROR "No c++11 or c++0x support in compiler."
            "You can not use -DNOTIFY_PROGRESS=ON without it"
            )
    ENDIF()
ENDIF()
################################### TARGETS ####################################

add_library(_moose MODULE pymoose/moosemodule.cpp)
add_library(libmoose SHARED basecode/main.cpp)
add_executable(moose.bin basecode/main.cpp)


################################### SETUP BUILD ################################


# If this variable is used while packaging; we build this project using another
# cmake script and pass the location of SBML static library using this variable.
# Otherwise, this is useless. FIXME: This is not a clean solution.

if(SBML_STATIC_HOME)
    include_directories(${SBML_STATIC_HOME}/include)
    find_library(LIBSBML_LIBRARIES
        NAMES libsbml-static.a libsbml.a
        PATHS "${SBML_STATIC_HOME}/lib" "${SBML_STATIC_HOME}/lib64"
        NO_DEFAULT_PATH
        )
    message(STATUS "- Using static version of LIBSBML_LIBRARIES: ${LIBSBML_LIBRARIES}")
    if(LIBSBML_LIBRARIES)
        message(STATUS "- Successfully located SBML static library: ${LIBSBML_LIBRARIES}")
    ELSE()
        message(FATAL_ERROR 
            "Can't find static libsbml libraries at path ${SBML_STATIC_HOME}"
            )
    endif()
    set(LIBSBML_FOUND ON)
else(SBML_STATIC_HOME)
    pkg_check_modules(LIBSBML libsbml)
    if(NOT LIBSBML_FOUND)
        message(STATUS "pkg-config could not find sbml. Fallback to default")
        find_package(LIBSBML)
    endif()
endif(SBML_STATIC_HOME)

if(LIBSBML_FOUND)
    message(STATUS "LIBSBML found ${LIBSBML_LIBRARIES}")
    include_directories(${LIBSBML_INCLUDE_DIRS})
    pkg_check_modules(LibXML2 libxml-2.0)
    if (!LibXML2_FOUND)
        find_package(LibXML2 REQUIRED)
    endif()
    include_directories(${LibXML2_INCLUDE_DIRS})
    if(${LIBSBML_LIBRARY_DIRS})
        target_link_libraries(libmoose PROPERTIES LINK_FLAGS
            "-L${LIBSBML_LIBRARY_DIRS}"
            )
        target_link_libraries(_moose PROPERTIES LINK_FLAGS
            "-L${LIBSBML_LIBRARY_DIRS}"
            )
        target_link_libraries(cymoose PROPERTIES LINK_FLAGS
            "-L${LIBSBML_LIBRARY_DIRS}"
            )
    endif()

else()
    message(
        "======================================================================\n"
        "libsbml NOT found. \n\n"
        "If you want to compile with SBML support, download and install \n"
        "libsbml-5.9.0 from: \n"
        "http://sourceforge.net/projects/sbml/files/libsbml/5.9.0/stable/ and\n"
        "rerun cmake.\n\n"
        "If you don't want SBML support then continue with `make`.\n\n"
        "If you install libsbml to non-standard place, let the cmake know by\n"
        "exporting environment variable SBML_DIR to the location.\n"
        "=====================================================================\n"
        )
    SET(WITH_SBML OFF)
endif()

include_directories(msg basecode)

set_target_properties(libmoose PROPERTIES COMPILE_DEFINITIONS  "MOOSE_LIB")
set_target_properties(libmoose PROPERTIES PREFIX "")

set(Boost_USE_STATIC_LIBS OFF)
find_package( Boost REQUIRED COMPONENTS system filesystem log REQUIRED )
find_package( LAPACK REQUIRED )
add_definitions( -DUSE_BOOST )

# When not using boost-static libraries, enable boost-log dynamic link macro.
if( NOT Boost_USE_STATIC_LIBS )
    add_definitions( -DBOOST_LOG_DYN_LINK )
endif()

include_directories( 
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/boost-numeric-bindings
    )

## Setup hdf5
if(HDF5_STATIC_HOME)
    SET(HDF5_INCLUDE_DIRS "${HDF5_STATIC_HOME}/include")
    find_library(HDF5_LIBRARY 
        NAME libhdf5-static.a
        PATHS ${HDF5_STATIC_HOME}/lib ${HDF5_STATIC_HOME}/lib64
        NO_DEFAULT_PATH
        )
    find_library(HDF5_HL_LIBRARY
        NAME libhdf5_hl-static.a
        PATHS ${HDF5_STATIC_HOME}/lib ${HDF5_STATIC_HOME}/lib64
        NO_DEFAULT_PATH
        )
    SET(HDF5_LIBRARIES ${HDF5_LIBRARY} ${HDF5_HL_LIBRARY})
    SET(HDF5_FOUND TRUE)
ELSE(HDF5_STATIC_HOME)
    pkg_check_modules(HDF5 hdf5)
    if(NOT HDF5_FOUND)
        message(STATUS "pkg-config could not find hdf5. fallback to default")
        find_package(HDF5 COMPONENTS CXX HL)
     endif()
endif(HDF5_STATIC_HOME)

if(NOT HDF5_FOUND)
    message(
        "==================================================================\n"
        " HDF5 not found. Disabling support. Required for nsdf. \n\n"
        " If you need hdf5 support, please install hdf5-dev or hdf5-devel\n"
        " package or equivalent.\n\n"
        "     $ sudo apt-get install libhdf5-dev \n"
        "     $ sudo yum install libhdf5-devel \n"
        "     $ brew install hdf5 \n\n"
        " Otherwise, continue with 'make' and 'make install' \n"
        " If you install hdf5 to non-standard path, export environment \n"
        " variable HDF5_ROOT to the location. Rerun cmake \n"
        "================================================================ \n"
        )
ELSEIF(HDF5_FOUND AND (NOT HDF5_STATIC_HOME))
    if(HDF5_HL_LIBRARIES)
        set(HDF5_LIBRARIES ${HDF5_HL_LIBRARIES})
    else()
        set(HDF5_LIBRARIES "-lhdf5 -lhdf5_hl")
    endif()
    include_directories(${HDF5_INCLUDE_DIRS})
endif(NOT HDF5_FOUND)

if(HDF5_FOUND)
    add_definitions( -DUSE_HDF5 )
    include_directories( ${HDF5_INCLUDE_DIRS} )
endif( HDF5_FOUND )

# This is a fix for new HDF5 package on Debian/Ubuntu which installs hdf5
# headers in non-standard path. issue #80.
if(HDF5_LIBRARY_DIRS)
    set_target_properties( libmoose PROPERTIES LINK_FLAGS "-L${HDF5_LIBRARY_DIRS}" )
    set_target_properties( _moose PROPERTIES LINK_FLAGS "-L${HDF5_LIBRARY_DIRS}" )
endif()

find_package(Termcap)
find_package(Readline)

if(READLINE_FOUND AND TERMCAP_FOUND)
    add_definitions(-DUSE_READLINE)
    include_directories(${Readline_INCLUDE_DIR})
endif()

# Openmpi
if(WITH_MPI)
    find_package(MPI REQUIRED)
    if(MPI_CXX_FOUND)
        message(STATUS "Using MPI from ${MPI_CXX_INCLUDE_PATH}")
        include_directories(${MPI_CXX_INCLUDE_PATH})
        set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
        add_definitions(-DUSE_MPI)
        SET(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
        SET(CMAKE_C_COMPILER ${MPI_C_COMPILER})
    else()
        message(STATUS "Cound not find MPI")
        add_definitions(-UUSE_MPI)
    endif()
endif(WITH_MPI)

# Add subdirectroeis
add_subdirectory(basecode)
add_subdirectory(msg)
add_subdirectory(shell)
add_subdirectory(scheduling)
add_subdirectory(biophysics)
add_subdirectory(builtins)
add_subdirectory(utility)
add_subdirectory(mesh)
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
add_subdirectory(external/muparser)

if(LIBSBML_FOUND)
    add_subdirectory(sbml)
endif(LIBSBML_FOUND)

# These are always shared libraries.
set(SYSTEM_SHARED_LIBS ${LibXML2_LIBRARIES})

# These libraries could be static of dynamic.
# When building packages on OBS or launchpad, we build static library of a
# dependencies if a correct version is not found. 
# Here we make sure that the correct static version is linked; if a correct
# shared version is found then link the shared version in SYSTEM_SHARED_LIBS.


if(HDF5_STATIC_HOME)
    list(APPEND STATIC_LIBRARIES ${HDF5_LIBRARIES})
elseif(HDF5_FOUND)
    list(APPEND SYSTEM_SHARED_LIBS ${HDF5_LIBRARIES})
endif()

if(LIBSBML_FOUND)
    LIST(APPEND STATIC_LIBRARIES moose_sbml)
endif(LIBSBML_FOUND)

if(SBML_STATIC_HOME)
    list(APPEND STATIC_LIBRARIES ${LIBSBML_LIBRARIES})
elseif(LIBSBML_FOUND)
    list(APPEND SYSTEM_SHARED_LIBS ${LIBSBML_LIBRARIES})
endif()

# When boost is used, we use lapack binding instead of gsl to compute steady
# state.
if(USE_BOOST)
    list( APPEND SYSTEM_SHARED_LIBS ${LAPACK_LIBRARIES} )
    list( APPEND SYSTEM_SHARED_LIBS ${Boost_LIBRARIES} )
endif(USE_BOOST)

if(READLINE_FOUND AND TERMCAP_FOUND)
    list(APPEND SYSTEM_SHARED_LIBS ${Readline_LIBRARY} ${TERMCAP_LIBRARY})
endif()

if(WITH_MPI)
    if(MPI_CXX_FOUND)
        list(APPEND SYSTEM_SHARED_LIBS ${MPI_CXX_LIBRARIES})
    endif()
endif(WITH_MPI)


#pkg_check_modules(TINYXML tinyxml)
#if(TINYXML_FOUND)
    #message(STATUS "Using system tinyxml ${TINYXML_VERSION}")
    #list(APPEND SYSTEM_SHARED_LIBS ${TINYXML_LIBRARIES})
#else()
    #message(STATUS "Using private tinyxml")
    #add_subdirectory(external/tinyxml)
    #list(APPEND MOOSE_LIBRARIES tinyxml)
#endif()

###################################### LINKING #################################
list(APPEND MOOSE_LIBRARIES
    moose_builtins
    msg
    benchmarks
    shell
    scheduling
    moose_mpi
    biophysics 
    utility 
    kinetics 
    synapse
    intfire
    hsolve 
    mesh
    signeur
    diffusion 
    ksolve
    device
    basecode
    muparser
    )


# MAC linker does not understand many of gnu-ld options.
message( STATUS "Shared libs: ${SYSTEM_SHARED_LIBS}")
if(MACOSX)
    target_link_libraries(libmoose
        "-Wl,-all_load"
        ${MOOSE_LIBRARIES}
        ${STATIC_LIBRARIES}
        )

    target_link_libraries(libmoose
        ${SYSTEM_SHARED_LIBS}
        )
ELSE(MACOSX)
    target_link_libraries(libmoose 
        "-Wl,--whole-archive"
        ${MOOSE_LIBRARIES}
        ${STATIC_LIBRARIES}
        "-Wl,--no-whole-archive"
        ${SYSTEM_SHARED_LIBS}
        )
endif(MACOSX)

add_dependencies(moose.bin libmoose)

# This target is only for testing purpose. It links to libmoose.
#if(CMAKE_VERSION VERSION_LESS "2.8.0")
#    message(STATUS "Using a fix for cmake ( < 2.8)")
#    target_link_libraries(moose.bin PUBLIC moose)
#ELSE()

target_link_libraries(moose.bin LINK_PUBLIC moose ${SYSTEM_SHARED_LIBS})

# ENDIF()

if(WITH_DOC)
    FIND_PACKAGE(Doxygen REQUIRED)
    add_custom_command(TARGET libmoose POST_BUILD
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.full
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building developer documentation"
        VERBATIM
        )
endif(WITH_DOC)

######################### BUILD PYMOOSE ########################################
# Root of all python module.
SET(PROJECT_PYTHON_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python)
if(WITH_PYTHON)
    message(STATUS "Building native python extension of MOOSE")

    set(PYMOOSE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/moose")

    find_package(NumPy REQUIRED)
    find_package(PythonDev REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    message(STATUS "Python include dir: ${PYTHON_INCLUDE_DIRS}")
    message(STATUS "Python libraries: ${PYTHON_LIBRARIES}")
    message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")

    EXEC_PROGRAM(${PYTHON_EXECUTABLE}
        ARGS "-c 'try: import importlib.machinery; print(importlib.machinery.EXTENSION_SUFFIXES[0])
except Exception: pass'"
        OUTPUT_VARIABLE PYTHON_SO_EXTENSION
        )

    add_subdirectory(pymoose)

    include_directories(${NUMPY_INCLUDE_DIRS})
    add_definitions(-DUSE_NUMPY)

    set_target_properties(_moose PROPERTIES COMPILE_DEFINITIONS "PYMOOSE")
    set_target_properties(_moose PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY ${PYMOOSE_OUTPUT_DIRECTORY}
        PREFIX "")
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
        moosepython  
        ${MOOSE_LIBRARIES}
        ${STATIC_LIBRARIES}
        )
    target_link_libraries(_moose
        ${SYSTEM_SHARED_LIBS}
        )
ELSE(MACOSX)
    target_link_libraries(_moose 
        "-Wl,--whole-archive"
        moosepython  
        ${MOOSE_LIBRARIES}
        ${STATIC_LIBRARIES}
        "-Wl,--no-whole-archive"
        ${PYTHON_LIBRARIES}
        ${SYSTEM_SHARED_LIBS}
        )
endif(MACOSX)

# Make this target dependant on all python files in python folder.
add_custom_target(copy_python_files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/python
    ${CMAKE_CURRENT_BINARY_DIR}/python
    COMMENT "Copying python files to build directory"
    VERBATIM
    )
add_dependencies(_moose copy_python_files)

if(WITH_DOC)
    # Also generate the documents
    FIND_PROGRAM(Sphinx REQUIRED)
    add_custom_command(TARGET _moose
        POST_BUILD
        COMMAND make html
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Docs/user/py
        COMMENT "Building user doc using python-sphinx"
        VERBATIM
        )
endif(WITH_DOC)
endif(WITH_PYTHON)

######################### CYMOOSE #############################################
option(BUILD_CYMOOSE "Build python extension using Cython. Development version." OFF)
if(BUILD_CYMOOSE)
    message(STATUS "Building experimental cython extension of MOOSE.")
    add_library(cymoose SHARED cymoose/cymoose_methods.cxx)
    set_target_properties(cymoose PROPERTIES COMPILE_DEFINITIONS "CYMOOSE")
    if(DEBUG)
        add_definitions("-g -Wall")
    endif()

    set_target_properties(cymoose PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cymoose"
            )

if(MACOSX)
    target_link_libraries(cymoose 
        "-Wl,-all_load"
        ${MOOSE_LIBRARIES}
        ${LIBRARIES}
        "-Wl,-noall_load"
        ${SYSTEM_LIBRARIES}
        )
ELSE(MACOSX)
    target_link_libraries(cymoose 
        "-Wl,--whole-archive"
        ${MOOSE_LIBRARIES}
        ${LIBRARIES}
        "-Wl,--no-whole-archive"
        ${SYSTEM_LIBRARIES}
        )
endif(MACOSX)

endif(BUILD_CYMOOSE)

## Moose documentation
option(WITH_DOC "Build documentation using python-sphinx and doxygen" OFF)
if(WITH_DOC)
    FIND_PACKAGE(Sphinx REQUIRED)
    message(STATUS "Build documentation.")
    set(USER_DOC_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Docs/user/py/_build/html)
    set(DEVELOPER_DOC_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Docs/developer/html)
    ADD_CUSTOM_TARGET(docs ALL
        COMMAND ./docgen
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating html doc using sphinx and doxygen"
        )

endif(WITH_DOC)

######################### INSTALL ##############################################

install(TARGETS moose.bin
    DESTINATION bin
    )

install(TARGETS libmoose
    DESTINATION lib
    )

install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/moose
    DESTINATION bin
    )

if(WITH_PYTHON)
    install(CODE 
        "
        execute_process(
            COMMAND ${PYTHON_EXECUTABLE} setup.py install --prefix=${CMAKE_INSTALL_PREFIX}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python
            )
        "
        )

endif(WITH_PYTHON)

if(WITH_DOC)
    message(STATUS "Installing moose doc")
    install(DIRECTORY ${USER_DOC_OUTPUT_DIR}
        DESTINATION share/doc/moose
        )

    install(DIRECTORY ${DEVELOPER_DOC_OUTPUT_DIR}
        DESTINATION share/doc/moose/developer
        )
endif()

# Print message to start build process
if(${CMAKE_BUILD_TOOL} MATCHES "make")
    message(
        "=======================================\n"
        "If cmake did not report any error, run \n"
	" 'make' to build  MOOSE \n"
        "=======================================\n"
        )
endif()


############################ CTEST ######################################

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake_moose_ctest.cmake)

############################ CPACK ######################################
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake_moose_cpack.cmake)
