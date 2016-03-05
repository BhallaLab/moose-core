# - Try to find Python include dirs and libraries
#
# Usage of this module as follows:
#
#     find_package(PythonDev)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  PYTHON_EXECUTABLE         If this is set to a path to a Python interpreter
#                            then this module attempts to infer the path to
#                            python-config from it
#  PYTHON_CONFIG             Set this variable to the location of python-config
#                            if the module has problems finding the proper
#                            installation path.
#
# Variables defined by this module:
#
#  PYTHONDEV_FOUND           System has Python dev headers/libraries
#  PYTHON_INCLUDE_DIRS        The Python include directories.
#  PYTHON_LIBRARIES          The Python libraries and linker flags.

include(FindPackageHandleStandardArgs)
include(FindPkgConfig)

if (CMAKE_SYSTEM_NAME STREQUAL Linux)
    if (!PYTHON_EXECUTABLE)
        find_program(PYTHON_EXECUTABLE NAMES python2 python python3)
    endif()
    EXEC_PROGRAM(${PYTHON_EXECUTABLE}
        ARGS "-c 'import sys;print(sys.version_info.major)'"
        OUTPUT_VARIABLE PYTHON_MAJOR_VERSION)
    pkg_check_modules(PYTHON REQUIRED "python${PYTHON_MAJOR_VERSION}")

else ()
    if (PYTHON_EXECUTABLE AND EXISTS ${PYTHON_EXECUTABLE}-config)
        set(PYTHON_CONFIG ${PYTHON_EXECUTABLE}-config CACHE PATH "" FORCE)
    else ()
        find_program(PYTHON_CONFIG
            NAMES python-config python-config2.7 python-config2.6 python-config2.6
                  python-config2.4 python-config2.3)
    endif ()

    # The OpenBSD python packages have python-config's that don't reliably
    # report linking flags that will work.
    if (PYTHON_CONFIG AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL "OpenBSD")
        execute_process(COMMAND "${PYTHON_CONFIG}" --ldflags
                        OUTPUT_VARIABLE PYTHON_LIBRARIES
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
        execute_process(COMMAND "${PYTHON_CONFIG}" --includes
                        OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)

        string(REGEX REPLACE "^[-I]" "" PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIRS}")
        string(REGEX REPLACE "[ ]-I" " " PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIRS}")
        separate_arguments(PYTHON_INCLUDE_DIRS)

        find_package_handle_standard_args(PythonDev DEFAULT_MSG
            PYTHON_CONFIG
            PYTHON_INCLUDE_DIRS
            PYTHON_LIBRARIES
        )
    else ()
        find_package(PythonLibs)
        if (PYTHON_INCLUDE_PATH AND NOT PYTHON_INCLUDE_DIRS)
            set(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_PATH}")
        endif ()
        find_package_handle_standard_args(PythonDev DEFAULT_MSG
            PYTHON_INCLUDE_DIRS
            PYTHON_LIBRARIES
        )
    endif ()

endif ()
