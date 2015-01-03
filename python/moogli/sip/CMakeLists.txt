cmake_minimum_required(VERSION 2.8)
FILE(GLOB SIP_SRC *.sip)

SET(GENERATED_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sip_mooglicmodule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sip_moogliMorphology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sip_moogliMorphologyViewer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sip_moogliSelectInfo.cpp
    )


MESSAGE("++ Using sip includes: ${SIP_INCLUDES}")
add_custom_command(OUTPUT ${GENERATED_SRCS}
    COMMAND ${SIP_EXECUTABLE} -c ${CMAKE_CURRENT_SOURCE_DIR} -I ${SIP_INCLUDES} 
        -e -g -x VendorID -t WS_X11 -x PyQt_NoPrintRangeBug -t Qt_4_8_0 -x Py_v3 -g
        ${CMAKE_CURRENT_SOURCE_DIR}/_moogli.sip
    DEPENDS ${SIP_SRC}
    COMMENT "Processing ${CMAKE_CURRENT_SOURCE_DIR}/_moogli.sip"
    VERBATIM
    )

ADD_LIBRARY(_moogli SHARED
    ${GENERATED_SRCS}
)

SET_TARGET_PROPERTIES(_moogli PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/moogli"
    PREFIX ""
    )

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
