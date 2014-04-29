cmake_minimum_required(VERSION 2.8)

project(moose)

# This snippet is from LLVM project.

# Sanity check our source directory to make sure that we are not trying to
# generate an in-tree build (unless on MSVC_IDE, where it is ok), and to make
# sure that we don't have any stray generated files lying around in the tree
# (which would end up getting picked up by header search, instead of the correct
# versions).
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE )
  message(FATAL_ERROR "In-source builds are not allowed.
CMake would overwrite the makefiles distributed with LLVM.
Please create a directory and run cmake from there, passing the path
to this source directory as the last argument.
This process created the file `CMakeCache.txt' and the directory `CMakeFiles'.
Please delete them.")
endif()
if( NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR )
  file(GLOB_RECURSE
    tablegenned_files_on_include_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/include/llvm/*.gen")
  file(GLOB_RECURSE
    tablegenned_files_on_lib_dir
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/Target/*.inc")
  if( tablegenned_files_on_include_dir OR tablegenned_files_on_lib_dir)
    message(FATAL_ERROR "Apparently there is a previous in-source build,
probably as the result of running `configure' and `make' on
${CMAKE_CURRENT_SOURCE_DIR}.
This may cause problems. The suspicious files are:
${tablegenned_files_on_lib_dir}
${tablegenned_files_on_include_dir}
Please clean the source directory.")
  endif()
endif()

# These are CMAKE macros. Must be on the top.
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
set(INSTALL_LIB_DIR lib CACHE PATH "/usr/local/lib")
set(INSTALL_BIN_DIR bin CACHE PATH "/usr/local/bin")
set(INSTALL_INCLUDE_DIR include CACHE PATH "/usr/local/include/")

# Do the gsl first
set(LOCAL_GSL 0)
if(LOCAL_GSL)
    add_subdirectory(gsl)
    include_directories(gsl)
else()
    find_package(GSL REQUIRED)
endif()

# Compiler 
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_COMPILER "g++")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_COMPILER "clang++")
else()
    message(WARNING "Continuing with unsupported compiler")
endif()

# Use clang++ as default compiler for time being
set(CMAKE_CXX_COMPILER "clang++")

## Turn warning to error
add_definitions(-Wall
    -Wno-return-type-c-linkage
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-unused-private-field
    -Werror
    )

## Enable 2011 stupport.
set(ENABLE_STD_2011 0)
if(ENABLE_STD_2011)
    add_definitions(-DENABLE_STD_2011 -DDEBUG2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif(ENABLE_STD_2011)

# Strick checking 
set(DEVELOPER 1)
if(DEVELOPER)
    MESSAGE("++ Enabled unit tests")
    add_definitions(-DDO_UNIT_TESTS -DENABLE_TESTS)
    add_definitions(-DDEBUG -DDEVELOPER -fPIC)
else()
    add_definitions(-UDO_UNIT_TESTS)
endif()

set(VERIFY 1)
if(VERIFY)
    add_definitions(-DSANITY_CHECK -DARGS_CHECK -DRESULT_CHECK -DVALUE_CHECK)
endif(VERIFY)

# Default macros
add_definitions(-DUSE_GENESIS_PARSER)

set(DEBUG 1)
if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
endif()

## Here we add an executable
## These two directories must be included for main.cpp 
include_directories(msg basecode)
add_executable(moose basecode/main.cpp)

## GSL
add_definitions(-DUSE_GSL)
if(LOCAL_GSL)
    include_directories(gsl)
else()
    include_directories(${GSL_INCLUDE_DIR})
    target_link_libraries(moose ${GSL_LIBRARIES})
endif()


## Setup hdf5
find_package(HDF5)
if(HDF5_FOUND)
    add_definitions(-DUSE_HDF5)
    include_directories(${HDF5_INCLUDE_DIR})
    target_link_libraries(moose ${HDF5_LIBRARY})
endif(HDF5_FOUND)

find_package(BZip2)
find_package(LIBSBML)
if(LIBSBML_FOUND)
    add_definitions(-DUSE_SBML)
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
    target_link_libraries(moose ${Readline_LIBRARY})
    target_link_libraries(moose ${TERMCAP_LIBRARY})
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
add_subdirectory(external/debug/)
add_subdirectory(mesh)
add_subdirectory(mpi)
#add_subdirectory(signeur)
add_subdirectory(ksolve)
add_subdirectory(hsolve)
add_subdirectory(diffusion)
add_subdirectory(device)
add_subdirectory(benchmarks)
add_subdirectory(kinetics)

# Links our libraries 
target_link_libraries(moose 
    scheduling
    builtins
    muparser
    device
    moose_mpi
    msg 
    diffusion
    #signeur
    mesh
    ksolve
    hsolve
    debug_print
    benchmarks
    shell
    )

if(LIBSBML_FOUND)
    target_link_libraries(moose moose_sbml ${LIBSBML_LIBRARY})
    if(BZIP2_FOUND)
        target_link_libraries(moose ${BZIP2_LIBRARIES})
    else()
        target_link_libraries(moose bz2)
    endif()
endif()

# Link gsl 
set(LOCAL_GSL_LIBRARIES  gsl gslcblas m)
target_link_libraries(moose ${LOCAL_GSL_LIBRARIES})

# Libxml2 is required.
find_package(LibXML2 REQUIRED)
target_link_libraries(moose ${LibXML2_LIBRARIES})

################################################################################
## PYTHON related stuff.
################################################################################

######################### BUILD PYMOOSE ########################################
set(BUILD_PYMOOSE 1)
if(BUILD_PYMOOSE)
    find_package(NumpyHeaders REQUIRED)
    if(PYTHON_NUMPY_FOUND)
        add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
        include_directories(${PYTHON_NUMPY_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "Could not find numpy/arrayobject.h in usual places")
    endif()
    # Openmpi
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})

    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    add_subdirectory(pymoose)
    add_library(_moose SHARED pymoose/moosemodule.cpp)
    #add_library(_moose SHARED pymoose/moosemodule.cpp)
    set_target_properties(_moose PROPERTIES COMPILE_DEFINITIONS "PYMOOSE")
    set_target_properties(_moose PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY "./python/moose"
        PREFIX ""
        )

    set(LIBRARIES ${PYTHON_LIBRARY} ${LibBZip2_LIBRARIES} 
        ${LOCAL_GSL_LIBRARIES}
        ${LibXML2_LIBRARIES}
        )
    if(HDF5_FOUND)
        list(APPEND LIBRARIES ${HDF5_LIBRARY})
    endif()
    if(LIBSBML_FOUND)
        list(APPEND LIBRARIES moose_sbml ${LIBSBML_LIBRARY})
    endif()

    target_link_libraries(_moose 
        "-Wl,--whole-archive"
        moosepython
        ksolve
        basecode
        msg
        scheduling
        #signeur
        shell
        randnum
        moose_mpi
        builtins
        utility
        diffusion
        hsolve
        muparser
        biophysics
        kinetics
        mesh
        debug_print
        device
        benchmarks
        pthread
        ${LIBRARIES}
        "-Wl,--no-whole-archive"
        )
endif(BUILD_PYMOOSE)

if(BUILD_PYMOOSE)
    # This macro compiles each python file and store them in BUILD_DIRECTORY
    macro(add_python_target tgt)
        foreach(file ${ARGN})
            GET_FILENAME_COMPONENT(FilePath ${file} PATH)
            GET_FILENAME_COMPONENT(FileNameWE ${file} NAME_WE)
            set(FileName "${FilePath}/${FileNameWE}.pyc")
            list(APPEND OUT_FILES ${FileName})
            add_custom_command(OUTPUT ${FileName}
                COMMAND python -m py_compile ${CMAKE_SOURCE_DIR}/${file}
                COMMAND ${CMAKE_COMMAND} -E copy 
                    ${CMAKE_SOURCE_DIR}/${FileName} ${CMAKE_BINARY_DIR}/${FileName})
        endforeach()

        add_custom_target(${tgt} ALL
            DEPENDS ${OUT_FILES}
            )
    endmacro()
    
    # Search files using glob
    FILE(GLOB_RECURSE Python_Src RELATIVE ${CMAKE_SOURCE_DIR} python/*.py)
    add_python_target(python_src_compiled
        ${Python_Src}
        )
endif()

################################################################################
## Keep CTEST here.
################################################################################
include(CTest)
find_package(PythonInterp REQUIRED)
add_test(NAME run_moose_with_enable_test
    COMMAND ./moose 
    )
add_test(NAME load_moose_into_python
    COMMAND ${PYTHON_EXECUTABLE} "python/moose/test.py"
    )
