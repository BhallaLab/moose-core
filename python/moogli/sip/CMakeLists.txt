cmake_minimum_required(VERSION 2.8)
FILE(GLOB SIP_SRC *.sip)

SET(GENERATED_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/sip_mooglicmodule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sip_moogliMorphology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sip_moogliMorphologyViewer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sip_moogliSelectInfo.cpp
    )


SET(SIP_INCLUDES ${PYQT4_SIP_DIR})
MESSAGE("++ Using sip includes: ${SIP_INCLUDES}")
add_custom_command(OUTPUT ${GENERATED_SRCS}
    COMMAND ${SIP_EXECUTABLE} -c ${CMAKE_CURRENT_SOURCE_DIR} -I ${SIP_INCLUDES} 
        ${PYQT4_SIP_FLAGS}
        ${CMAKE_CURRENT_SOURCE_DIR}/_moogli.sip
    DEPENDS ${SIP_SRC}
    COMMENT "Processing ${CMAKE_CURRENT_SOURCE_DIR}/_moogli.sip"
    VERBATIM
    )

ADD_LIBRARY(sip  ${GENERATED_SRCS})


