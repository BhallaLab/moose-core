cmake_minimum_required(VERSION 2.8)
SET(CMAKE_VERBOSE_MAKEFILE ON)

MESSAGE("++ SIP_INCLUDES ${SIP_INCLUDES}")
include_directories(${SIP_INCLUDES})

ADD_SIP_PYTHON_MODULE(libmoogli _moogli.sip)
ADD_SIP_PYTHON_MODULE(libmorp Morphology.sip)
ADD_SIP_PYTHON_MODULE(libmorphview MorphologyViewer.sip)
