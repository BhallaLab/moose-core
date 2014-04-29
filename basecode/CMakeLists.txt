cmake_minimum_required(VERSION 2.8)
include_directories(../msg ../external/debug)
add_library(basecode 
    Element.cpp
    DataElement.cpp
    GlobalDataElement.cpp
    LocalDataElement.cpp
    Cinfo.cpp
    consts.cpp
    DataElement.cpp
    DestFinfo.cpp
    doubleEq.cpp
    EpFunc.cpp
    Eref.cpp
    FieldElement.cpp
    FieldElementFinfo.cpp
    Finfo.cpp
    HopFunc.cpp
    Id.cpp
    ObjId.cpp
    OpFuncBase.cpp
    SetGet.cpp
    SharedFinfo.cpp
    SparseMatrix.cpp
    PrepackedBuffer.cpp
    SrcFinfo.cpp
    testAsync.cpp
    ValueFinfo.cpp
    global.cpp
    )
target_link_libraries(basecode debug_print biophysics)
