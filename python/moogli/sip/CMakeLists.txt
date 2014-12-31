cmake_minimum_required(VERSION 2.8)
SET(CMAKE_VERBOSE_MAKEFILE ON)

MESSAGE("++ SIP_INCLUDES ${SIP_INCLUDES}")
include_directories(${SIP_INCLUDES})

ADD_SIP_PYTHON_MODULE(_moogli _moogli.sip)
ADD_SIP_PYTHON_MODULE(morp Morphology.sip)
ADD_SIP_PYTHON_MODULE(morphview MorphologyViewer.sip)
