cmake_minimum_required(VERSION 2.8)
include( ${CMAKE_CURRENT_SOURCE_DIR}/../CheckCXXCompiler.cmake )

set(SRCS
    Arith.cpp
    Group.cpp
    Mstring.cpp
    Func.cpp
    Function.cpp
    Variable.cpp
    InputVariable.cpp
    TableBase.cpp
    Table.cpp
    Interpol.cpp
    StimulusTable.cpp
    TimeTable.cpp
    StreamerBase.cpp
    Streamer.cpp
    Stats.cpp
    Interpol2D.cpp
    SpikeStats.cpp
    testBuiltins.cpp
    )

if(WITH_TCP_STREAMER)
    set(TCP_SOCKET_PORT 31416)
    message(STATUS "TCP Streamer enabled. It will stream data on port"
                    " ${TCP_SOCKET_PORT} (remember pi)")
    add_definitions(-DTCP_SOCKET_PORT=${TCP_SOCKET_PORT})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../external/asio-1.12.2/include)
    list(APPEND SRCS SocketStreamer.cpp)
endif(WITH_TCP_STREAMER)

if(WITH_NSDF AND HDF5_FOUND)
    list(APPEND SRCS
        HDF5WriterBase.cpp
        NSDFWriter.cpp
        HDF5DataWriter.cpp
        testNSDF.cpp
        )
endif()

add_library(moose_builtins ${SRCS} )
