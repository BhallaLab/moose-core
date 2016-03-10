ENABLE_TESTING()

FIND_PACKAGE(PythonInterp REQUIRED)

# If CTEST_OUTPUT_ON_FAILURE environment variable is set, the output is printed
# onto the console if a test fails.
SET(ENV{CTEST_OUTPUT_ON_FAILURE} ON)

ADD_TEST(NAME moose.bin-raw-run
    COMMAND moose.bin -u -q
    )

## PyMOOSE tests.

SET(PYMOOSE_TEST_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/python)

# This test does not work with python-2.6 because of unittest changed API.
#ADD_TEST(NAME pymoose-test-pymoose
#    COMMAND ${PYTHON_EXECUTABLE} test_pymoose.py
#    WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY}
#    )

ADD_TEST(NAME pymoose-test-synchan
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/python/test_synchan.py
    )
set_tests_properties(pymoose-test-synchan PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python")

ADD_TEST(NAME pymoose-test-function
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/python/test_function.py
    )
set_tests_properties(pymoose-test-function PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python")

ADD_TEST(NAME pymoose-test-vec
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/python/test_vec.py
    )
set_tests_properties(pymoose-test-vec PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python")

ADD_TEST(NAME pymoose-pyrun
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/python/test_pyrun.py
    )
set_tests_properties(pymoose-pyrun PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python")

# Do not run this test after packaging.
ADD_TEST(NAME pymoose-neuroml-reader-test 
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/python/test_neuroml.py
    )
set_tests_properties(pymoose-neuroml-reader-test PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python")

ADD_TEST(NAME pymoose-nsdf-sanity-test
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/python/test_nsdf.py
    )
set_tests_properties(pymoose-nsdf-sanity-test PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python")

# Test basic SBML support.
ADD_TEST(NAME pymoose-test-basic-sbml-support
    COMMAND ${PYTHON_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/tests/python/test_sbml_support.py
    )
set_tests_properties(pymoose-test-basic-sbml-support 
    PROPERTIES ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python"
    )


##IF(WITH_SBML)
##    ADD_TEST(NAME pymoose-test-sbml 
##        COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/python/test_sbml.py
##        )
##ENDIF(WITH_SBML)

##ADD_TEST(NAME pymoose-test-kkit 
##    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/python/test_kkit.py
##    )
