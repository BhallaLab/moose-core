# CMake script to install python.
if(CPACK_TEMPORARY_INSTALL_DIR)
    MESSAGE("++ Using cpack to generate package")
    set(CMAKE_INSTALL_PREFIX ${CPACK_TEMPORARY_INSTALL_DIR})
endif()
execute_process(COMMAND 
    python -c "from distutils.sysconfig import get_python_lib; print get_python_lib(prefix=\"${CMAKE_INSTALL_PREFIX}\")" 
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
    )
MESSAGE("++ Installing python scripts: ${CMAKE_INSTALL_PREFIX}")
FILE(MAKE_DIRECTORY ${PYTHON_SITE_PACKAGES})
MESSAGE("+++ Updating PYTHONPATH: ${PYTHON_SITE_PACKAGES}")
set(ENV{PYTHONPATH} ${PYTHON_SITE_PACKAGES})
set(PYTHON_DIR ${CMAKE_CURRENT_LIST_DIR})
execute_process(COMMAND 
    python setup.py build_py build -b /tmp install
        --prefix=${CMAKE_INSTALL_PREFIX} 
    WORKING_DIRECTORY ${PYTHON_DIR}
    )
FILE(REMOVE_RECURSE ${PYTHON_DIR}/build)
