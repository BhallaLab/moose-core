# distutils: language = c++
# distutils: extra_compile_args = -DCYMOOSE

cimport Eref as _Eref
cimport Compartment as _Compartment
cimport Id as _Id

# Class 
cdef class PyCompartment:

    cdef _Compartment.Compartment* thisptr
    cdef public PyId obj
    cdef _Eref.Eref* pEref

    def __init__(self, obj):
        self.obj = PyId()

    def __cinit__(self):
        self.thisptr = new _Compartment.Compartment()

    def __deinit__(self):
        del self.thisptr


    cdef double getVm(self, _Eref.Eref& e):
        return self.thisptr.vGetVm(e)

