# language: c++

from libcpp.string cimport string
from cython.operator import dereference

include "PyId.pyx"

cimport Shell as _Shell
cimport Id as _Id
cimport ObjId as _ObjId


from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "./cymoose.hpp":
    _Shell.Shell* initMoose(int argc, char** argv, _Shell.Shell* s)

cdef class PyShell:

    cdef _Shell.Shell *thisptr

    def __cinit__(self):
        cdef char* argv = ""
        self.thisptr = initMoose(0, &argv, self.thisptr)

    def __dealloc__(self):
        del self.thisptr 


    def create(self, string elemType, string elemPath, unsigned int numData
            , _Shell.NodePolicy nodePolicy = _Shell.MooseBlockBalance
            , unsigned int preferredNode = 1):
        cdef _Id.Id obj
        obj = self.thisptr.create(elemType, elemPath , numData, nodePolicy, preferredNode)
        newObj = PyId()
        newObj.thisptr = &obj
        newObj.setPath()
        return newObj
