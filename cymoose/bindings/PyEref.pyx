# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

cimport bindings.Eref as _Eref

cdef class PyEref:

    cdef _Eref.Eref* thisptr

    def __cinit__(self):
        self.thisptr = new _Eref.Eref()

    def __del__(self):
        del self.thisptr
