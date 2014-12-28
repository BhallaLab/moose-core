# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

cimport Id as _Id
cimport ObjId as _ObjId 
cimport Neutral as _Neutral
cimport Compartment as _Compartment
cimport Shell as _Shell
#from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

include "PyId.pyx"
include "PyShell.pyx"
include "PyCompartment.pyx"
include "PyEref.pyx"
include "PyObjId.pyx"

shell = PyShell()


## CyMoose functions

def wildcardFind(pattern):
    cdef vector[_ObjId.ObjId] paths
    cdef int ret = _Shell.wildcardFind(pattern, paths)
    pypath = []
    for p in paths:
        obj = PyObjId()
        obj.objId = p
        pypath.append(obj)
    return pypath

cdef class Neutral:
    """Neutral class """

    cdef public PyId obj

    def __cinit__(self, path):
        self.obj =  shell.create("Neutral", path, 1)

    def __repr__(self):
        return "Id: {0}, Type: Neutral, Path: {1}".format(None, self.obj.path)

    def path(self):
        return self.obj.path


cdef class Compartment:

    """ Compartment class """

    cdef public PyId obj
    cdef public PyCompartment comp_ 

    def __cinit__(self, path, numData = 1):
        self.obj = shell.create("Compartment", path, numData)
        # Wrap this id in python compartment
        self.comp_ = PyCompartment(self.obj)

    def __repr__(self):
        return "Id: {0}, Type: Compartment, Path: {1}".format(1, self.obj.path)

    def path(self):
        return self.obj.path

    property Vm:
        def __get__(self):
            return self.comp_.getVm(deref(self.obj.eref_.thisptr))

        def __set__(self, value):
            self.comp_.setVm(self.obj.eref_, value)

