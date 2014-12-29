# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON


cimport bindings.Neutral as _neutral
cimport bindings.Eref as _eref

cdef class PyNeutral:
    cdef _neutral.Neutral *thisptr
    def __cinit__(self):
        self.thisptr = new _neutral.Neutral()
    def __dealloc__(self):
        del self.thisptr



