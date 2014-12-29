# distutils: language = c++
# distutils: include_dirs = ., cymoose, ..
# distutils: extra_compile_args = -DCYTHON


"""compartment.pyx: 

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, Dilawar Singh"
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

from modules.shell import shell_
from bindings.PyId import PyId
from bindings.PyCompartment import PyCompartment

from cython.operators import dereference as deref

cdef class Compartment:

    """ Compartment class """

    cdef public object obj
    cdef public object comp_ 
    cdef public object compartments

    def __cinit__(self, path, numData = 1):
        # Following creates as many elements as given by numData. This function
        # returns only top-object. We should iterate over it and create a list
        # of objects created.
        self.obj = shell_.create("Compartment", path, numData)

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
            self.comp_.setVm(deref(self.obj.eref_.thisptr), value)

