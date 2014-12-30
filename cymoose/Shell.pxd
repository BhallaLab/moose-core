# distutils: language = c++
# distutils: include_dirs = ., cymoose
# distutils: extra_compile_args = -DCYTHON

cimport Id as _Id
cimport ObjId as _ObjId
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "../shell/Shell.h":
    cdef enum NodePolicy:
       MooseGlobal
       MooseBlockBalance 
       MooseSingleNode

    cdef enum AssignmentType:
       SINGLE
       VECTOR
       REPEAT

    cdef extern cppclass Shell:
        Shell() 
        Shell* initShell()
        _Id.Id create(string elemType
                , string name
                , unsigned int numData 
                , NodePolicy nodePolicy 
                , unsigned int preferredNode 
                ) 
        bool doDelete(_ObjId.ObjId id)

        # Function to  connect.
        _ObjId.ObjId doAddMsg( const string& msgType, 
                _ObjId.ObjId src, const string& srcField, 
                _ObjId.ObjId dest, const string& destField )

        

cdef extern from "../shell/Wildcard.h":
    int wildcardFind(const string& n, vector[_ObjId.ObjId]& ret) 

