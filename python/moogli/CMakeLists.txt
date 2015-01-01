CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
# setup SIP variables
SEPARATE_ARGUMENTS(PYQT4_SIP_FLAGS) # convert space separated values to a list
SET(SIP_INCLUDES ${PYQT4_SIP_DIR})
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
ADD_LIBRARY(_moogli SHARED
    ${SIP_TO_CPP_SOURCES}
)

add_dependencies(_moogli mooglisip)

SET_TARGET_PROPERTIES(_moogli PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/moogli"
    PREFIX ""
    )

add_subdirectory(moc)
add_subdirectory(src)

target_link_libraries(_moogli
    moc
    src
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
