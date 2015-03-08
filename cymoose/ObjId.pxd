# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

from libcpp.string cimport string 
cimport Id as _Id

cdef extern from "../basecode/ObjId.h":
    cdef extern cppclass ObjId:
        ObjId() 
        ObjId(_Id.Id id)
        ObjId(_Id.Id id, unsigned int dataIndex, unsigned int fieldIndex)
        string path()
