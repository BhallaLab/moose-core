cimport Neutral as _neutral
cimport Eref as _eref

cdef class Neutral:
    cdef _neutral.Neutral *thisptr
    def __cinit__(self):
        self.thisptr = new _neutral.Neutral()
    def __dealloc__(self):
        del self.thisptr



