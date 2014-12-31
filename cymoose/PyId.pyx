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
    cdef public unsigned int id_
    cdef public string str_

    def __cinit__(self
            , unsigned int id = 0
            , string path = ""
            ):
        if id != 0:
            self.thisptr = new _Id.Id(id)
        elif len(path) > 0:
            self.thisptr = new _Id.Id(path)
        else:
            self.thisptr = new _Id.Id()

        self.thisptr.value()

        self.objId = PyObjId()
        self.objId.pId = self.thisptr

    def __repr__(self):
        return "Id: {}".format(self.value())

    def __del__(self):
        del self.thisptr

    cdef unsigned int value(self):
        self.id_ = self.thisptr.value()
        return self.id_

    cdef string id2str(self, _Id.Id id_):
        self.str_ = self.thisptr.id2str(id_)
        return self.str_

    def setPath(self, sep = "/"):
        self.path = self.thisptr.path(sep)

    def eref(self):
        erefObj = PyEref()
        cdef _Eref.Eref e = self.thisptr.eref()
        erefObj.thisptr = &e
        self.eref_ = erefObj
        return erefObj
