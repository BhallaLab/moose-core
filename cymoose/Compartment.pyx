# distutils: language = c++
# distutils: extra_compile_args = -DCYTHON

cimport Eref as _Eref
cimport Compartment as _Compartment

# Class 
cdef class PyCompartment:

    cdef _Compartment.Compartment* thisptr

    def __cinit__(self):
        self.thisptr = new _Compartment.Compartment()

    def __deinit__(self):
        del self.thisptr

    #property Vm:
        #def __get__(self, e): return self.thisptr.vGetVm(e)
        #def __set__(self, e, x): self.thisptr.vSetVm(e, x)

    #property Em:
        #def __get__(self, e) : return self.thisptr.vGetEm(e)
        #def __set__(self, e, x): self.thisptr.vSetEm(e, x)
    
    #property Rm:
        #def __get__(self, e): return self.thisptr.vGetRm(e)
        #def __set__(self, e, x): self.thisptr.vSetRm(e, x)
    
    #property Ra:
        #def __get__(self, e): return self.thisptr.vGetRa(e)
        #def __set__(self, e, x): self.thisptr.vSetRa(e, x)

    #property Im:
        #def __get__(self, e): return self.thisptr.vGetIm(e)
        #def __set__(self, e, x): self.thisptr.vSetIm(e, x)

    #property Inject:
        #def __get__(self, e): return self.thisptr.vGetInject(e)
        #def __set__(self, e, x): self.thisptr.vSetInject(e, x)

    #property initVm:
        #def __get__(self, e): return self.thisptr.vGetInitVm(e)
        #def __set__(self, e, x): self.thisptr.vSetInitVm(e, x)

    #property lenght:
        #def __get__(self, e): return self.thisptr.vGetLength(e)
        #def __set__(self, e, x): self.thisptr.vSetLength(e, x)


