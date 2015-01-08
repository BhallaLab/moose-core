cmake_minimum_required(VERSION 2.8)
FILE(GLOB SIP_SRC *.sip)

SET(GENERATED_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sipmooglicmodule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sipmoogliMorphology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sipmoogliMorphologyViewer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sipmoogliSelectInfo.cpp
    )

SET(SIP_INCLUDES ${PYQT4_SIP_DIR})

MESSAGE("++ Using sip includes: ${SIP_INCLUDES}")
add_custom_command(OUTPUT ${GENERATED_SRCS}
    COMMAND ${SIP_EXECUTABLE} -c ${CMAKE_CURRENT_SOURCE_DIR} -I ${SIP_INCLUDES} 
        ${SIP_EXTRA_OPTIONS}
        ${CMAKE_CURRENT_SOURCE_DIR}/moogli.sip
    DEPENDS ${SIP_SRC}
    COMMENT "Processing ${CMAKE_CURRENT_SOURCE_DIR}/_moogli.sip"
    VERBATIM
    )

ADD_LIBRARY(sip  ${GENERATED_SRCS})
