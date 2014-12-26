cimport Id as _Id 

cdef class Id:
    cdef _Id.Id* thisptr

    def __init__(self):
        self.thisptr = new _Id.Id()

