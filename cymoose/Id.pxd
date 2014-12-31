# Wrapper around basecode/Id.h files 

cimport Eref as _Eref
cimport ObjId as _ObjId
from libcpp.string cimport string 

cdef extern from "../basecode/Id.h":
    cdef extern cppclass Id:
        Id() 
        Id(unsigned int id)
        Id(_ObjId.ObjId& objid)
        Id(string path, string sep = "/")
        string path(string)
        _Eref.Eref eref()
        unsigned int value()
        string id2str( Id id_)
