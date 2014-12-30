# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

import sys

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
        obj.thisptr = &p
        pypath.append(obj)
    return pypath


def connect(string msgType, srcObj, string srcField
        ,  destObj, string destField):
    if type(srcObj) == Compartment:
        shell.connect(msgType, srcObj.obj_, srcField, destObj.obj_, destField)
    sys.exit()
    return shell.connect(msgType, srcObj, srcField, destObj, destField)



cdef class Neutral:
    """Neutral class """

    cdef public PyId id_
    cdef public PyObjId obj_

    def __cinit__(self, path):
        self.id_ =  shell.create("Neutral", path, 1)
        self.obj_ = PyObjId(self.id_)

    def __repr__(self):
        return "Id: {0}, Type: Neutral, Path: {1}".format(None, self.id_.path)

    def path(self):
        return self.id_.path


cdef class Compartment:

    """ Compartment class """

    cdef public PyId id_
    cdef public PyObjId obj_
    cdef public PyCompartment comp_ 
    cdef public object compartments
    cdef _Eref.Eref* objEref 

    def __cinit__(self, path, numData = 1):
        # Following creates as many elements as given by numData. This function
        # returns only top-object. We should iterate over it and create a list
        # of objects created.
        self.id_ = shell.create("Compartment", path, numData)
        self.obj_ = PyObjId(self.id_)

        # Wrap this id in python compartment
        self.comp_ = PyCompartment(self.id_)
        self.objEref = self.id_.eref_.thisptr

    def __repr__(self):
        return "Id: {0}, Type: Compartment, Path: {1}".format(1, self.id_.path)

    def path(self):
        return self.obj.path

    property Vm:
        def __get__(self):
            return self.comp_.getVm(deref(self.objEref))
        def __set__(self, value):
            self.comp_.setVm(deref(self.objEref), value)

