# CMake script to install python.
MESSAGE("Instlling python scripts: ${CMAKE_INSTALL_PREFIX}")
set(PYTHON_DIR ${CMAKE_CURRENT_LIST_DIR})
execute_process(COMMAND python setup.py build_py install --prefix=${CMAKE_INSTALL_PREFIX} 
    WORKING_DIRECTORY ${PYTHON_DIR}
    )
FILE(REMOVE_RECURSE ${PYTHON_DIR}/build)
