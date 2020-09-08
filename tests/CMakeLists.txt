set(CTEST_NIGHTLY_START_TIME "05:30:00 UTC")
set(CTEST_SUBMIT_URL "http://my.cdash.org/submit.php?project=moose")

if(DEBUG OR "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    # Run this test in debug mode. In Release mode, this does not do anything.
    add_test(NAME moose.bin-raw-run COMMAND $<TARGET_FILE:moose.bin> -u -q)
endif()

# Core tests.
set(PYMOOSE_TEST_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/core)
file(GLOB PY_TEST_SCRIPTS "${PYMOOSE_TEST_DIRECTORY}/test_*.py" )
foreach(_test_script ${PY_TEST_SCRIPTS} )
    get_filename_component(_name_we ${_test_script} NAME_WE)
    set(_test_name "core_${_name_we}")
    add_test(NAME ${_test_name}
        COMMAND ${PYTHON_EXECUTABLE} ${_test_script}
        WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY})
     set_tests_properties(${_test_name}
         PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
        )
endforeach()

# Tests for supported formats such as neuroml, sbml etc.
set(SUPPORT_TEST_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/support)
file(GLOB PY_TEST_SCRIPTS "${SUPPORT_TEST_DIRECTORY}/test_*.py" )
foreach(_test_script ${PY_TEST_SCRIPTS} )
    get_filename_component(_name_we ${_test_script} NAME_WE)
    set(_test_name "support_${_name_we}")
    add_test(NAME ${_test_name}
        COMMAND ${PYTHON_EXECUTABLE} ${_test_script}
        WORKING_DIRECTORY ${SUPPORT_TEST_DIRECTORY})
     set_tests_properties(${_test_name}
         PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
        )
endforeach()

# rdesigneur tests. These tests require matplotlib.
set(RDES_TEST_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rdesigneur) 
file(GLOB RDES_TEST_SCRIPTS "${RDES_TEST_DIRECTORY}/test_*.py" )
foreach(_test_script ${RDES_TEST_SCRIPTS})
    get_filename_component(_name_we ${_test_script} NAME_WE)
    set(_test_name "rdes_${_name_we}")
    add_test(NAME ${_test_name}
        COMMAND ${PYTHON_EXECUTABLE} ${_test_script}
        WORKING_DIRECTORY ${RDES_TEST_DIRECTORY}
        )
    set_tests_properties(${_test_name} PROPERTIES 
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python;MOOSE_NUM_THREADS=4"
        )
endforeach()

# FIXME TESTS. These should not run by default. We need to fix them.
set(PYMOOSE_FIXME_TEST_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/devel/fixme)
file(GLOB PY_FIXME_TEST_SCRIPTS "${PYMOOSE_FIXME_TEST_DIRECTORY}/*.py" )
foreach( _test_script ${PY_FIXME_TEST_SCRIPTS} )
    get_filename_component( _name_we ${_test_script} NAME_WE)
    set(_test_name "alpha_${_name_we}")
    add_test( NAME ${_test_name}
        COMMAND ${PYTHON_EXECUTABLE} ${_test_script}
        CONFIGURATIONS alpha
	WORKING_DIRECTORY ${PYMOOSE_ALPHA_TEST_DIRECTORY}
        )
     set_tests_properties( ${_test_name}
         PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
        )
endforeach( )

# Regression and github issues. These should not run by default.
set(PYMOOSE_ISSUES_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/devel/issues)
file(GLOB PY_ISSUES_SCRIPTS "${PYMOOSE_ISSUES_DIRECTORY}/*.py" )
foreach(_test_script ${PY_ISSUES_SCRIPTS})
    get_filename_component(_file_name ${_test_script} NAME_WE)
    set(_test_name "issue_${_file_name}")
    add_test(NAME ${_test_name}
        COMMAND ${PYTHON_EXECUTABLE} ${_test_script}
	CONFIGURATIONS Devel
	WORKING_DIRECTORY ${PYMOOSE_ISSUES_DIRECTORY}
        )
     set_tests_properties(${_test_name}
         PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
        )
endforeach()

################################# COVERAGE ###################################

add_custom_target(coverage)
set(PYTEST_TEST_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
file(GLOB_RECURSE PYTEST_TEST_SCRIPTS ${PYTEST_TEST_DIRECTORY}/test_*.py)

foreach(_test_script ${PYTEST_TEST_SCRIPTS})
    file(MD5 ${_test_script} _test_hash)
    set(_test_name "coverage_${_test_hash}")
    # message(STATUS "Adding test ${_test_name} ${_test_hash}")
    add_custom_target(${_test_name}
        # We collect coverage data but do not show report during each test.
        # We show report at the end of all tests. Set `--cov-report=` 
        COMMAND ${PYTHON_EXECUTABLE} -m pytest 
            --cov=moose --cov=rdesigneur --cov-append --cov-report= 
            ${_test_script}
        DEPENDS _moose
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Executing ${_test_script} using pytest"
        VERBATIM)
    add_dependencies(coverage ${_test_name})
endforeach()

# Generate report at the end.
add_custom_command(TARGET coverage POST_BUILD
    COMMAND ${PYTHON_EXECUTABLE} -m coverage report 
    COMMAND ${PYTHON_EXECUTABLE} -m coverage html 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Genearting coverage report"
    VERBATIM)

