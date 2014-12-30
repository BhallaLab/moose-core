# distutils: language = c++

from libcpp.string cimport string
from cython.operator import dereference as deref

cimport Shell as _Shell
cimport Id as _Id
cimport ObjId as _ObjId


from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "./cymoose.hpp":
    _Shell.Shell* initMoose(int argc, char** argv, _Shell.Shell* s)

cdef class PyShell:

    cdef _Shell.Shell *thisptr

    def __cinit__(self):
        cdef char* argv = ""
        self.thisptr = initMoose(0, &argv, self.thisptr)

    def __dealloc__(self):
        del self.thisptr 


    def create(self, string elemType, string elemPath, unsigned int numData
            , _Shell.NodePolicy nodePolicy = _Shell.MooseBlockBalance
            , unsigned int preferredNode = 1):
        """Call shell create function """
        cdef _Id.Id obj
        obj = self.thisptr.create(elemType, elemPath , numData, nodePolicy, preferredNode)
        newObj = PyId()
        newObj.thisptr = &obj
        newObj.setPath()
        return newObj


    # Function delete clashes with built-in name, therefore erase is provided.
    cdef erase(self, _ObjId.ObjId objId):
        self.thisptr.doDelete(objId)
        

    # This function expose toAddMsg function
    cdef _ObjId.ObjId add_msg(self, string msgType
            , _ObjId.ObjId srcObj, const string& srcField
            , _ObjId.ObjId destObj, const string& destField
            ):
        """Call shell do addMsg function """
        cdef _ObjId.ObjId objId 
        return self.thisptr.doAddMsg(msgType, srcObj, srcField
                , destObj, destField
                )

    def connect(self, string msgType, PyObjId srcObj, string srcField
            , PyObjId destObj, string destField
            ):
        cdef _ObjId.ObjId objId 
        objId = self.add_msg(msgType, deref(srcObj.thisptr), srcField
                , deref(destObj.thisptr), destField)
        obj = PyObjId()
        obj.thisptr = &objId 
        return obj


