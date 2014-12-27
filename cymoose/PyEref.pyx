cimport Eref as _Eref

cdef class PyEref:

    cdef _Eref.Eref* thisptr

    def __cinit__(self):
        self.thisptr = new _Eref.Eref()

    def __del__(self):
        del self.thisptr
