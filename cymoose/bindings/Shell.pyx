# Wrapper around msg/Shell class

from libcpp.string cimport string

cimport bindings.Shell as _Shell
cimport bindings.Id as _id 
cimport bindings.ObjId as _objid 

cdef class PyShell:

    cdef _Shell.Shell *thisptr

    def __cinit__(self):
        self.thisptr = new _Shell.Shell()

    def __dealloc__(self):
        del self.thisptr 

    cdef _id.Id create(self
            , _type
            ,  _name
            , _numData
            , _nodePolicy
            , _preferedNode
            ):
        return self.thisptr.create(_type, _name, _numData
                , _nodePolicy , _preferedNode)
    
    def callCreate(self):
        print("Here writing a wrapper function.")
