cmake_minimum_required(VERSION 2.8)

SET(CMAKE_VERBOSE_MAKEFILE ON)

add_custom_command(OUTPUT
    ${CMAKE_CURRENT_SOURCE_DIR}/_moogli.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Morphology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MorphologyViewer.cpp
    COMMAND ${SIP_EXECUTABLE} -c ${CMAKE_CURRENT_SOURCE_DIR} -I ${SIP_INCLUDES} 
    -e -g -x VendorID -t WS_X11 -x PyQt_NoPrintRangeBug -t Qt_4_8_0 -x Py_v3 -g
    ${CMAKE_CURRENT_SOURCE_DIR}/_moogli.sip
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/_moogli.sip
    COMMENT "Processing ${CMAKE_CURRENT_SOURCE_DIR}/_moogli.sip"
    VERBATIM
    )

add_custom_target(mooglisip ALL DEPENDS _moogli.sip)
set(SIP_TO_CPP_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/sip_mooglicmodule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sip_moogliMorphology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sip_moogliMorphologyViewer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sip_moogliSelectInfo.cpp
    )

