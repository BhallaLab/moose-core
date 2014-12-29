# distutils: language = c++
# distutils: extra_compile_args = -DCYTHON

cimport bindings.Eref as _Eref
cimport bindings.CompartmentBase as _CompartmentBase

# Class 
cdef class PyCompartmentBase:
    cdef _CompartmentBase.CompartmentBase* thisptr

    def __cinit__(self):
        self.thisptr = new _CompartmentBase.CompartmentBase()

    def __deinit__(self):
        del self.thisptr

    #property Vm:
        #def __get__(self): return self.thisptr.getVm(e)
        #def __set__(self, x): self.thisptr.setVm(x)

    #property Em:
        #def __get__(self): return self.thisptr.getEm()
        #def __set__(self, x): self.thisptr.setEm(x)
    
    #property Rm:
        #def __get__(self): return self.thisptr.getRm()
        #def __set__(self, x): self.thisptr.setRm(x)
    
    #property Ra:
        #def __get__(self): return self.thisptr.getRa()
        #def __set__(self, x): self.thisptr.setRa(x)

    #property Im:
        #def __get__(self): return self.thisptr.getIm()
        #def __set__(self, x): self.thisptr.setIm(x)

    #property Inject:
        #def __get__(self): return self.thisptr.getInject()
        #def __set__(self, x): self.thisptr.setInject(x)

    #property initVm:
        #def __get__(self): return self.thisptr.getInitVm()
        #def __set__(self, x): self.thisptr.setInitVm(x)

    #property diameter:
        #def __get__(self): return self.thisptr.getDiameter()
        #def __set__(self, x): self.thisptr.setDiameter(x)

    #property lenght:
        #def __get__(self): return self.thisptr.getLength()
        #def __set__(self, x): self.thisptr.setLength(x)

