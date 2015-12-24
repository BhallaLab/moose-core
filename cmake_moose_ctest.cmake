ENABLE_TESTING()

FIND_PACKAGE(PythonInterp REQUIRED)

# If CTEST_OUTPUT_ON_FAILURE environment variable is set, the output is printed
# onto the console if a test fails.
SET(ENV{CTEST_OUTPUT_ON_FAILURE} ON)

ADD_TEST(NAME moose.bin-raw-run
    COMMAND moose.bin -u -q
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

## PyMOOSE tests.

SET(PYMOOSE_TEST_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/python)

# This test does not work with python-2.6 because of unittest changed API.
#ADD_TEST(NAME pymoose-test-pymoose
#    COMMAND ${PYTHON_EXECUTABLE} test_pymoose.py
#    WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY}
#    )

ADD_TEST(NAME pymoose-test-synchan
    COMMAND ${PYTHON_EXECUTABLE} test_synchan.py
    WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY}
    )

ADD_TEST(NAME pymoose-test-function
    COMMAND ${PYTHON_EXECUTABLE} test_function.py
    WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY}
    )

ADD_TEST(NAME pymoose-test-vec
    COMMAND ${PYTHON_EXECUTABLE} test_vec.py
    WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY}
    )

ADD_TEST(NAME pymoose-pyrun
    COMMAND ${PYTHON_EXECUTABLE} test_pyrun.py
    WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY}
    )

# Do not run this test after packaging.
ADD_TEST(NAME pymoose-neuroml-reader-test 
    COMMAND ${PYTHON_EXECUTABLE} test_neuroml.py
    WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY}
    )

ADD_TEST(NAME pymoose-nsdf-sanity-test
    COMMAND ${PYTHON_EXECUTABLE} test_nsdf.py
    WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY}
    )

##IF(WITH_SBML)
##    ADD_TEST(NAME pymoose-test-sbml 
##        COMMAND ${PYTHON_EXECUTABLE} test_sbml.py
##        WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY}
##        )
##ENDIF(WITH_SBML)

##ADD_TEST(NAME pymoose-test-kkit 
##    COMMAND ${PYTHON_EXECUTABLE} test_kkit.py
##    WORKING_DIRECTORY ${PYMOOSE_TEST_DIRECTORY}
##    )

