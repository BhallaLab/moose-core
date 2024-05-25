cmake_minimum_required(VERSION 3.20)
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

# Workaround for getopt included via unistd.h on Unices
if(WIN32)
target_include_directories(basecode PRIVATE ../extern/getopt)
target_link_libraries(basecode PRIVATE getopt)
endif()

add_executable(test_globals testGlobals.cpp global.cpp)

enable_testing()
add_test(NAME cpp_test_globals  COMMAND $<TARGET_FILE:test_globals>)

# DEBUG compilation process
get_target_property(compile_defs basecode COMPILE_DEFINITIONS)
message(STATUS "App compile definitions are ${compile_defs}")
# DEBUG compilation process

