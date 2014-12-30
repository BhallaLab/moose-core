# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

cimport Id as _Id 
cimport ObjId as _ObjId
cimport Eref as _Eref

from libcpp.string cimport string

from cython.operator import dereference

cdef class PyId:

    cdef _Id.Id* thisptr

    cdef public PyEref eref_
    cdef public string path
    cdef public PyObjId objId

    def __cinit__(self
            , unsigned int id = 0
            , string path = ""
            ):
        if id:
            self.thisptr = new _Id.Id(id)
        elif len(path) > 0:
            self.thisptr = new _Id.Id(path)
        else:
            self.thisptr = new _Id.Id()

        self.objId = PyObjId()
        self.objId.pId = self.thisptr

    def __del__(self):
        del self.thisptr

    def setPath(self, sep = "/"):
        self.path = self.thisptr.path(sep)

    def eref(self):
        erefObj = PyEref()
        cdef _Eref.Eref e = self.thisptr.eref()
        erefObj.thisptr = &e
        self.eref_ = erefObj
        return erefObj
