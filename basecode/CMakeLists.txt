cmake_minimum_required(VERSION 2.8)
include( ${CMAKE_CURRENT_SOURCE_DIR}/../CheckCXXCompiler.cmake )
add_library(basecode
	Element.cpp
	DataElement.cpp
	GlobalDataElement.cpp
	LocalDataElement.cpp
	Eref.cpp
	Finfo.cpp
	DestFinfo.cpp
	Cinfo.cpp
	SrcFinfo.cpp
	ValueFinfo.cpp
	SharedFinfo.cpp
	FieldElementFinfo.cpp
	FieldElement.cpp
	Id.cpp
	ObjId.cpp
	global.cpp
	SetGet.cpp
	OpFuncBase.cpp
	EpFunc.cpp
	HopFunc.cpp
	SparseMatrix.cpp
	doubleEq.cpp
	testAsync.cpp
    )

add_executable(test_globals testGlobals.cpp global.cpp)

enable_testing()
add_test(NAME cpp_test_globals  COMMAND $<TARGET_FILE:test_globals>)

