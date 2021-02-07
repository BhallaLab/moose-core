cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Project to build MOOSE's python module.
project(PyMOOSE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# cmake related macros.
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# We find python executable here. Though mainly used inside pymoose. 
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(CMAKE_MACOSX_RPATH OFF)

# NOTE: version should be changed in setup.py file.
# If moose version is not given, use setup.py file to get the default version.
if(NOT VERSION_MOOSE)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} setup.py --version
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE VERSION_MOOSE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

add_definitions(-DMOOSE_VERSION="${VERSION_MOOSE}")
message(STATUS "MOOSE Version ${VERSION_MOOSE}")

# This snippet is from LLVM project.
# Sanity check our source directory to make sure that we are not trying to
# generate an in-tree build (unless on MSVC_IDE, where it is ok), and to make
# sure that we don't have any stray generated files lying around in the tree
# (which would end up getting picked up by header search, instead of the correct
# versions).
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

################################ CMAKE OPTIONS ##################################
option(DEBUG                "Build with debug support" OFF)
option(GPROF                "Build for profiling using gprof" OFF)
option(ENABLE_UNIT_TESTS    "Enable unit tests (DEBUG should also be ON)" OFF)
option(WITH_MPI             "Enable Openmpi support" OFF)
option(WITH_ASAN            "Use AddressSanitizer in DEBUG mode." OFF)
option(WITH_NSDF            "Enable NSDF support. Requires hdf5" OFF )
option(WITH_LEGACY_BINDING  "Use legacy python-bindings" OFF)

############################ BUILD CONFIGURATION #################################

# At least version 1.53 (odeint is part of this version)
find_package(Boost 1.53 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Default definitions.
add_definitions(-DUSE_GENESIS_PARSER)
if(DEBUG OR "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Building for Debug/Unit testing")
    add_definitions(-DDO_UNIT_TESTS -O)
    set(CMAKE_BUILD_TYPE Debug)
elseif(ENABLE_UNIT_TESTS)
    MESSAGE(STATUS "Enabled Unit tests")
    add_definitions(-DDO_UNIT_TESTS)
    set(CMAKE_BUILD_TYPE Debug)
else()
    message(STATUS "Building for Release/No unit tests.")
    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-UDO_UNIT_TESTS -O3 -DDISABLE_DEBUG)
endif()

if(GPROF AND "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Compiling with profiling with gprof")
    add_definitions(-pg)
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-pg")
endif()

if(WITH_ASAN AND "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Compiling with ASAN")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
        -fno-omit-frame-pointer -fsanitize=leak -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} \
        -fno-omit-frame-pointer -fsanitize=leak -fsanitize=address")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address")
endif()

################################### TARGETS ####################################

link_directories(${CMAKE_BINARY_DIR})
add_library(libmoose SHARED basecode/main.cpp)
set_target_properties(libmoose PROPERTIES PREFIX "")
add_executable(moose.bin basecode/main.cpp)


################################### SETUP BUILD ################################

# Variable to collect all static libraries.
set(STATIC_LIBRARIES "" )
# Collect all shared libraries here.
set(SYSTEM_SHARED_LIBS "")


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

if(WITH_BOOST_ODE)
    list(APPEND SYSTEM_SHARED_LIBS ${LAPACK_LIBRARIES})
    list(APPEND SYSTEM_SHARED_LIBS ${Boost_LIBRARIES})
endif(WITH_BOOST_ODE)

find_package( Threads )
list(APPEND SYSTEM_SHARED_LIBS ${CMAKE_THREAD_LIBS_INIT})

# These libraries could be static of dynamic. We need to discrimate between
# these two types because of --whole-archive option. See
# BhallaLab/moose-core#66,

if(WITH_MPI)
    if(MPI_CXX_FOUND)
        list(APPEND SYSTEM_SHARED_LIBS ${MPI_CXX_LIBRARIES})
    endif()
endif(WITH_MPI)

# Add subdirectroeis
add_subdirectory(basecode)
add_subdirectory(msg)
add_subdirectory(shell)
add_subdirectory(randnum)
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
add_subdirectory(kinetics)
add_subdirectory(synapse)
add_subdirectory(intfire)
add_subdirectory(external)

# development related.
add_subdirectory(devel)

###################################### LINKING #################################
list(APPEND MOOSE_LIBRARIES
    moose_builtins
    msg
    shell
    randnum
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
    lsoda
    device
    basecode
    )

# Make sure to remove duplicates.
list(REMOVE_DUPLICATES STATIC_LIBRARIES)
if(SYSTEM_SHARED_LIBS)
    list(REMOVE_DUPLICATES SYSTEM_SHARED_LIBS)
endif( )

# cmake --help-policy CMP0042. Also in pymoose/CMakeLists.txt
# More details:
# https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling
# especially section 'Mac OS X and the RPATH'
# Switching is OFF since all libraries are statically linked in module.
if(APPLE)
    set_target_properties(libmoose PROPERTIES MACOSX_RPATH OFF)
endif(APPLE)

# MAC linker does not understand many of gnu-ld options.
# message( DEBUG "Shared libs: ${SYSTEM_SHARED_LIBS}")
if(APPLE)
    target_link_libraries(libmoose
        "-Wl,-all_load"
        ${MOOSE_LIBRARIES}
        ${STATIC_LIBRARIES}
        )

    target_link_libraries(libmoose
        ${SYSTEM_SHARED_LIBS}
        ${CMAKE_DL_LIBS}
        )
else(APPLE)
    target_link_libraries(libmoose
        "-Wl,--whole-archive"
        ${MOOSE_LIBRARIES}
        ${STATIC_LIBRARIES}
        "-Wl,--no-whole-archive"
        ${SYSTEM_SHARED_LIBS}
        )
endif(APPLE)

add_dependencies(moose.bin libmoose)
target_link_libraries(moose.bin moose ${CMAKE_DL_LIBS})

if( WITH_BOOST )
    target_link_libraries( moose.bin ${Boost_LIBRARIES} )
endif( WITH_BOOST )

######################### BUILD PYMOOSE ########################################

if(NOT WITH_LEGACY_BINDING)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11)
    add_subdirectory(pybind11)
else()
    message(STATUS "Building legacy python binding.")
    add_subdirectory(pymoose)
endif()


# always override debian default installation directory. It will be installed in
# site-packages instead of dist-packages.
# See https://bugs.launchpad.net/ubuntu/+source/python2.6/+bug/362570
# HACK: Get platform information from python and use it to fix the layout.
execute_process(COMMAND ${PYTHON_EXECUTABLE} -mplatform OUTPUT_VARIABLE _platform_desc)
message(STATUS "Platform: ${_platform_desc}")

# DISTUTILS_EXTRA_ARGS may come of top-level cmake script. On DEBIAN/UBUNTU, it
# is most likely to be --install-layout=deb .
set(EXTRA_ARGS "--prefix ${CMAKE_INSTALL_PREFIX} ${DISTUTILS_EXTRA_ARGS}")

# On Debian/Ubuntu install using debian layout.
# NOTE: Also create setup.cfg file which setup prefix and install-layout
# suitable for DEBIAN systems.
if(${_platform_desc} MATCHES ".*(Ubuntu|debian).*")
    list(APPEND EXTRA_ARGS "--install-layout=deb")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/setup.cfg
         "[install]\nprefix=/usr\ninstall-layout=deb"
       )
endif()


######################### INSTALL ##############################################

install(TARGETS moose.bin DESTINATION bin CONFIGURATIONS Debug)
install(TARGETS libmoose DESTINATION lib CONFIGURATIONS Debug)

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
include(CTest)
add_subdirectory(tests)

########################### RELEASE #########################################
set(PYMOOSE_SDIST_FILE ${CMAKE_BINARY_DIR}/pymoose-${VERSION_MOOSE}.tar.gz)
add_custom_target(sdist
    COMMAND ${PYTHON_EXECUTABLE} setup.py sdist --formats=gztar 
        -d ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Creating sdist ${PYMOOSE_SDIST_FILE}" 
    VERBATIM)

add_custom_target(upload_test DEPENDS sdist
    COMMAND ${PYTHON_EXECUTABLE} -m twine upload 
        --user $ENV{PYMOOSE_PYPI_USER} --password $ENV{PYMOOSE_PYPI_PASSWORD}
        --repository-url https://test.pypi.org/legacy/
        ${PYMOOSE_SDIST_FILE}
    COMMENT "Uploading source distribution to test.pypi.org"
    VERBATIM)

add_custom_target(upload DEPENDS sdist
    COMMAND ${PYTHON_EXECUTABLE} -m twine upload 
        --user $ENV{PYMOOSE_PYPI_USER} --password $ENV{PYMOOSE_PYPI_PASSWORD}
        ${PYMOOSE_SDIST_FILE}
    COMMENT "Uploading source distribution to PyPI"
    VERBATIM)

####################### DEVELOPMENT ########################################

# PYLINT target.
set(PYSCRIPTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/python)
file(GLOB_RECURSE PY_SCRIPTS  "python/*.py")
add_custom_target(pylint)
foreach(_py_script ${PY_SCRIPTS})
    get_filename_component( _py_name ${_py_script} NAME_WE)
    file( READ ${_py_script} pytext)
    string(MD5 _md5 "${pytext}")
    set(TGT_NAME "${_py_name}-${_md5}" )
    set(PYLINT_OPTIONS --disable=no-member --disable=no-name-in-module
            --disable=invalid-unary-operand-type
            --disable=import-error
            --disable=no-method-argument
        )
    add_custom_target( ${TGT_NAME}
        COMMAND ${PYTHON_EXECUTABLE} -m pylint -E ${PYLINT_OPTIONS} ${_py_script}
        COMMENT "Running pylint on ${_py_script}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    add_dependencies(pylint ${TGT_NAME} )
endforeach( )

########################  DOCS ###############################################
find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/devel/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation using Doxygen."
        VERBATIM)
else()
    message(STATUS "Doxygen needs to be installed to generate API docs")
endif()
