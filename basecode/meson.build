# Author: Subhasis Ray
# Date: Sun Jul  7

basecode_src = ['Element.cpp',
	        'DataElement.cpp',
	        'GlobalDataElement.cpp',
	        'LocalDataElement.cpp',
	        'Eref.cpp',
	        'Finfo.cpp',
	        'DestFinfo.cpp',
	        'Cinfo.cpp',
	        'SrcFinfo.cpp',
	        'ValueFinfo.cpp',
	        'SharedFinfo.cpp',
	        'FieldElementFinfo.cpp',
	        'FieldElement.cpp',
	        'Id.cpp',
	        'ObjId.cpp',
	        'global.cpp',
	        'SetGet.cpp',
	        'OpFuncBase.cpp',
	        'EpFunc.cpp',
	        'HopFunc.cpp',
	        'SparseMatrix.cpp',
	        'doubleEq.cpp',
	        'testAsync.cpp']

# On windows we provide getopt
if is_windows
  basecode_src += '../external/getopt/getopt.c'
  basecode_lib = static_library('basecode', basecode_src,
                                include_directories: '../external/getopt')
else
  basecode_lib = static_library('basecode', basecode_src)
endif
