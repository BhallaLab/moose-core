# languiage = C++

cimport Eref as _Eref
cimport Compartment as _Compartment

# Class 
cdef class PyCompartment:
    cdef _Compartment.Compartment* thisptr

    def __cinit__(self):
        self.thisptr = new _Compartment.Compartment()

    def __deinit__(self):
        del self.thisptr

    property Vm:
        def __get__(self): return self.thisptr.vGetVm(e)
        def __set__(self, x): self.thisptr.vSetVm(x)

    property Em:
        def __get__(self): return self.thisptr.vGetEm()
        def __set__(self, x): self.thisptr.vSetEm(x)
    
    property Rm:
        def __get__(self): return self.thisptr.vGetRm()
        def __set__(self, x): self.thisptr.vSetRm(x)
    
    property Ra:
        def __get__(self): return self.thisptr.vGetRa()
        def __set__(self, x): self.thisptr.vSetRa(x)

    property Im:
        def __get__(self): return self.thisptr.vGetIm()
        def __set__(self, x): self.thisptr.vSetIm(x)

    property Inject:
        def __get__(self): return self.thisptr.vGetInject()
        def __set__(self, x): self.thisptr.vSetInject(x)

    property initVm:
        def __get__(self): return self.thisptr.vGetInitVm()
        def __set__(self, x): self.thisptr.vSetInitVm(x)

    property diameter:
        def __get__(self): return self.thisptr.vGetDiameter()
        def __set__(self, x): self.thisptr.vSetDiameter(x)

    property lenght:
        def __get__(self): return self.thisptr.vGetLength()
        def __set__(self, x): self.thisptr.vSetLength(x)


