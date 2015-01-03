CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
# setup SIP variables
PROJECT(MOOGLI)

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake_modules")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_COMPILER "clang++")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_COMPILER "g++")
else()
    message(WARNING "Continuing with unsupported compiler: ${CMAKE_CXX_COMPILER}")
endif()

## Enable/Disable 2011 stupport.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

ADD_DEFINITIONS("-fPIC")

FIND_PACKAGE(PyQt4 REQUIRED)
SET(SIP_INCLUDES ${PYQT4_SIP_DIR})
MESSAGE("++ SIP_INCLUDES ${SIP_INCLUDES}")

FIND_PACKAGE(PythonLibs REQUIRED)
FIND_PACKAGE(PythonInterp REQUIRED)

SEPARATE_ARGUMENTS(PYQT4_SIP_FLAGS) # convert space separated values to a list
SET(SIP_CONCAT_PARTS 4)
SET(SIP_EXTRA_OPTIONS ${PYQT4_SIP_FLAGS})
IF (NOT BINDINGS_GLOBAL_INSTALL)
    SET(PYTHON_SITE_PACKAGES_DIR ${QGIS_DATA_DIR}/python)
ENDIF (NOT BINDINGS_GLOBAL_INSTALL)
IF (WITH_CUSTOM_WIDGETS)
    SET(PYUIC_WIDGET_PLUGIN_DIRECTORY ${PYQT4_MOD_DIR}/uic/widget-plugins/)
ENDIF (WITH_CUSTOM_WIDGETS)

FIND_PACKAGE(Qt4 REQUIRED COMPONENTS QtCore QtGui QtOpenGL)
FIND_PACKAGE(OpenSceneGraph REQUIRED)
FIND_PACKAGE(osgAnimation REQUIRED)

INCLUDE(${QT_USE_FILE})

FIND_PROGRAM(SIP_EXECUTABLE sip)
MESSAGE("++ SIP ${SIP_EXECUTABLE}")

include_directories(./include ${CMAKE_SOURCE_DIR} ".")
include_directories(${OSGQT_INCLUDE_DIR})
include_directories(${SIP_INCLUDES} ${PYTHON_INCLUDE_DIRS})
include_directories(${OPENSCENEGRAPH_INCLUDE_DIRS})
include_directories(${OSGANIMATION_INCLUDE_DIR})

add_subdirectory(sip)
add_subdirectory(moc)
add_subdirectory(src)

ADD_LIBRARY(_moogli SHARED 
    ${EMPTY_SOURCE}
    )

SET_TARGET_PROPERTIES(_moogli PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}"
    PREFIX ""
    )

## Link the following libraries to _moogli.
TARGET_LINK_LIBRARIES(_moogli
    moc
    src
    sip
    QtCore
    QtGui
    QtOpenGL           
    osg
    osgFX
    osgUtil
    osgFX
    osgGA
    osgQt
    osgAnimation
    osgViewer
    osgQt
    osgManipulator
    osgText
    )

## INSTALL components.
INSTALL(TARGETS _moogli
    LIBRARY DESTINATION share/moogli
    )

INSTALL(FILES ${PROJECT_SOURCE_DIR}/setup-moogli.py
    DESTINATION share/moogli
    )

SET(CPACK_STRIP_FILES TRUE)
SET(CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE TRUE)


INCLUDE(CPack)
