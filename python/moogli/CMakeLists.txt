CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)
LINK_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}/sip")
PROJECT(moogli)

add_definitions("-fPIC -std=c++0x")

SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(WITH_SIP 1)
IF (WITH_SIP)
    # python support: check for interpreter, sip, pyqt4
    FIND_PACKAGE(PythonInterp REQUIRED)
    FIND_PACKAGE(PythonLibs REQUIRED)
    FIND_PACKAGE(SIP REQUIRED)
    FIND_PACKAGE(PyQt4 REQUIRED)
    FIND_PACKAGE(osgQt REQUIRED)
    #INCLUDE(PythonMacros)
    INCLUDE(SIPMacros)
    INCLUDE(PyQt4Macros)

    # setup SIP variables
    SEPARATE_ARGUMENTS(PYQT4_SIP_FLAGS) # convert space separated values to a list
    SET(SIP_INCLUDES ${PYQT4_SIP_DIR} ${PYTHON_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/python)
    SET(SIP_CONCAT_PARTS 4)
    SET(SIP_EXTRA_OPTIONS ${PYQT4_SIP_FLAGS})
    IF (NOT BINDINGS_GLOBAL_INSTALL)
        SET(PYTHON_SITE_PACKAGES_DIR ${QGIS_DATA_DIR}/python)
    ENDIF (NOT BINDINGS_GLOBAL_INSTALL)
    IF (WITH_CUSTOM_WIDGETS)
        SET(PYUIC_WIDGET_PLUGIN_DIRECTORY ${PYQT4_MOD_DIR}/uic/widget-plugins/)
    ENDIF (WITH_CUSTOM_WIDGETS)
ENDIF (WITH_SIP)

FIND_PACKAGE( Qt4  REQUIRED COMPONENTS QtCore QtGui QtOpenGL )
FIND_PACKAGE(OpenSceneGraph REQUIRED)
FIND_PACKAGE(osgAnimation REQUIRED)

INCLUDE( ${QT_USE_FILE} )

include_directories(./include ${CMAKE_SOURCE_DIR})
include_directories(${OSGQT_INCLUDE_DIR})
include_directories(${SIP_INCLUDES})
include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
include_directories(${OSGANIMATION_INCLUDE_DIR})

ADD_LIBRARY(_moogli SHARED
    ./scenegraph.cpp
)

SET_TARGET_PROPERTIES(_moogli PROPERTIES PREFIX "")
add_subdirectory(moc)
add_subdirectory(sip)
add_subdirectory(src)

target_link_libraries(_moogli
    moc
    src
    ${OPENSCENEGRAPH_LIBRARIES}
    ${OSGANIMATION_LIBRARY}
    moogli
    morph
    morphview
    )

