cmake_minimum_required(VERSION 3.20)

add_library(msg
    Msg.cpp
    DiagonalMsg.cpp
    OneToAllMsg.cpp
    OneToOneMsg.cpp
    SingleMsg.cpp
    SparseMsg.cpp
    OneToOneDataIndexMsg.cpp
    testMsg.cpp
    )
