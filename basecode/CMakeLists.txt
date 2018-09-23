cmake_minimum_required(VERSION 2.8)
include( ${CMAKE_CURRENT_SOURCE_DIR}/../CheckCXXCompiler.cmake )
add_library(basecode
	consts.cpp
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

# tidy up 
if( "${CMAKE_VERSION}" VERSION_GREATER "3.5")
    set_target_properties(basecode 
        PROPERTIES  CXX_STANDARD 11 
                    CXX_STANDARD_REQUIRED ON 
                    COMPILE_FLAGS "${WARNING_FLAGS}"
        )
    if(clang_tidy_exe)
        set_target_properties(basecode 
            PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
            )
    endif()
endif()
