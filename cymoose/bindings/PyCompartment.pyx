# distutils: language = c++
# distutils: extra_compile_args = -DCYMOOSE

cimport bindings.Eref as _Eref
cimport bindings.Compartment as _Compartment
cimport bindings.Id as _Id

# Class 
cdef class PyCompartment:

    cdef _Compartment.Compartment* thisptr
    cdef public object obj
    cdef _Eref.Eref* pEref

    def __init__(self, obj):
        self.obj = PyId()

    def __cinit__(self):
        self.thisptr = new _Compartment.Compartment()

    def __deinit__(self):
        del self.thisptr

    cdef setVm(self, _Eref.Eref& e, double vm):
        self.thisptr.vSetVm(e, vm)

    cdef getVm(self, _Eref.Eref& e):
        self.thisptr.vGetVm(e)
