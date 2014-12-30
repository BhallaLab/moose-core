# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

# Definition to this class.

cimport ObjId as _ObjId
cimport Id as _Id
from libcpp.string cimport string

from cython.operator import dereference as deref
from libc.stdio cimport printf

cdef class PyObjId:

    cdef _ObjId.ObjId* thisptr

    cdef _Id.Id* pId 

    cdef PyId id_
    cdef public string path_

    def __repr__(self):
        msg = "<type: ObjId> at {}"
        if len(self.path_) < 1:
            self.path_ = self.path
        return msg.format(self.path_)

    def __cinit__(self, id_ = None, dataIndex = 0, fieldIndex = 0):
        self.thisptr = new _ObjId.ObjId()
        if id_:
            self.id_ = id_

    property path:
        def __get__(self): 
            if len(self.path_) > 0:
                return self.path_
            else:
                return self.id_.path

    property id_:
        def __get__(self): return self.id_ 
        def __set__(self, id_): self.id_ = id_

