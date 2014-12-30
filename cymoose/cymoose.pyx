# distutils: language = c++
# distutils: include_dirs = ., .., ../.., ../cymoose
# distutils: extra_compile_args = -DCYTHON
# distutils: libraries = cymoose

cimport bindings.Id as _Id
cimport bindings.ObjId as _ObjId 
cimport bindings.Neutral as _Neutral
cimport bindings.Compartment as _Compartment
cimport bindings.Shell as _Shell

#from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref


include "bindings/PyObjId.pyx"
include "bindings/PyShell.pyx"

import shell as shell
import compartment as comp

shell_ = shell.PyShell()

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

    cdef public object obj

    def __cinit__(self, path):
        self.obj =  shell_.create("Neutral", path, 1)

    def __repr__(self):
        return "Id: {0}, Type: Neutral, Path: {1}".format(None, self.obj.path)

    def path(self):
        return self.obj.path


