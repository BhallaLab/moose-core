# Definition to this class.
cimport ObjId as _ObjId

cdef class ObjId:

    cdef _ObjId.ObjId* thisptr

    def __init__(self):
        self.thisptr = new _ObjId.ObjId()
