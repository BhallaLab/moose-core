# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

# Definition to this class.
cimport ObjId as _ObjId

cdef class PyObjId:

    cdef _ObjId.ObjId* thisptr

    def __init__(self):
        self.thisptr = new _ObjId.ObjId()
