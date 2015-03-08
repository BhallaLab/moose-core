# distutils: language = c++
# distutils: extra_compile_args = -DCYMOOSE

cimport Eref as _Eref
cimport Compartment as _Compartment
cimport Id as _Id

from libcpp.map cimport map

# Class 
cdef class PyCompartment:

    cdef _Compartment.Compartment* thisptr
    cdef public PyId id_
    cdef public PyObjId obj_

    cdef _Eref.Eref* pEref

    def __init__(self, obj):
        self.id_ = obj
        self.obj_ = PyObjId(obj)

    def __cinit__(self):
        self.thisptr = new _Compartment.Compartment()

    def __deinit__(self):
        del self.thisptr

    cdef set(self, map[string, string] m):
        pass

    cdef setVm(self, _Eref.Eref& e, double vm):
        self.thisptr.vSetVm(e, vm)

    cdef double getVm(self, _Eref.Eref& e):
        return self.thisptr.vGetVm(e)

    
    
