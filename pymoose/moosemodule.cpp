// moosemodule.cpp --- 
// 
// Filename: moosemodule.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Thu Mar 10 11:26:00 2011 (+0530)
// Version: 
// Last-Updated: Thu Mar 31 15:30:20 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 3577
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 2011-03-10 Trying out direct access to Python API instead of going
//            via SWIG. SWIG has this issue of creating huge files and
//            the resulting binaries are also very large. Since we are
//            not going to use any language but Python in the
//            foreseeable future, we could as well try to avoid the
//            bloat by coding directly with Python API.
//

// TODO:
//
// Port the low level API.
//
// May use the low level API in Python to create the high level API.
//
// Allow exceptions in MOOSE. 

// Change log:
// 
// 2011-03-10 Initial version. Starting coding directly with Python API.
// 

// Code:

#include <Python.h>
#include <map>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <iostream>

#include "../basecode/header.h"
#include "../basecode/Id.h"
#include "../basecode/ObjId.h"
#include "../basecode/DataId.h"
#include "../utility/strutil.h"
#include "../utility/utility.h"
#include "../basecode/ReduceBase.h"
#include "../basecode/ReduceMax.h"
#include "../shell/Shell.h"

#include "moosemodule.h"

using namespace std;

extern const map<string, string>& getArgMap();
extern Shell& getShell();
extern void finalize();
extern void setup_runtime_env(bool verbose);
extern string getFieldType(ObjId id, string fieldName);
extern vector<string> getFieldNames(ObjId id, string fieldType);


extern int isSingleThreaded;
extern int isInfinite;
extern int numNodes;
extern int numCores;
extern int myNode;


// 
// C wrappers for C++ classes
// This is used by Python
extern "C" {
    static PyMethodDef IdMethods[] = {
        // {"init", (PyCFunction)_pymoose_Id_init, METH_VARARGS,
        //  "Initialize a Id object."},
        {"destroy", (PyCFunction)_pymoose_Id_destroy, METH_VARARGS,
         "destroy the underlying moose element"},
        {"getId", (PyCFunction)_pymoose_Id_getId, METH_VARARGS,
         "return integer representation of the id of the element."},
        {"syncDataHandler", (PyCFunction)_pymoose_Id_syncDataHandler, METH_VARARGS,
         "?"},
        {"getPath", (PyCFunction)_pymoose_Id_getPath, METH_VARARGS,
         "The path of this Id object."},
        {"getShape", (PyCFunction)_pymoose_Id_getShape, METH_VARARGS,
         "Get the shape of the Id object as a tuple."},
        {NULL, NULL, 0, NULL},        /* Sentinel */        
    };

    static PySequenceMethods IdSequenceMethods = {
        (lenfunc)_pymoose_Id_getLength, // sq_length
        0, //sq_concat
        0, //sq_repeat
        (ssizeargfunc)_pymoose_Id_getItem, //sq_item
        (ssizessizeargfunc)_pymoose_Id_getSlice, // getslice
        0, //sq_ass_item
        0, // setslice
        (objobjproc)_pymoose_Id_contains, // sq_contains
        0, // sq_inplace_concat
        0 // sq_inplace_repeat
    };

    // static PyGetSetDef Id_getsetters[] = {
    //     {"path",
    //      (getter)_pymoose_Id_getPath, NULL,
    //      "path of the element in the MOOSE object tree.",
    //      NULL},
    //     {"dimensions",
    //      (getter)_pymoose_Id_getDimensions, NULL,
    //      "dimensions of the object as tuple.",
    //     NULL},
    //     {"shape",
    //      (getter)_pymoose_Id_getDimensions, NULL,
    //      "dimensions of the object as tuple.",
    //     NULL},
    //     {"className",
    //      (getter)_pymoose_Id_getClass, NULL,
    //      "MOOSE class name of this object.",
    //     NULL}
    //     {"fieldNames",
    //      (getter)_pymoose_Id_getField, NULL,
        
    //     {NULL},
    // };
    static PyMethodDef ObjIdMethods[] = {
        {"getFieldType", (PyCFunction)_pymoose_ObjId_getFieldType, METH_VARARGS,
         "Get the string representation of the type of this field."},        
        {"getField", (PyCFunction)_pymoose_ObjId_getField, METH_VARARGS,
         "Get specified attribute of the element."},
        {"setField", (PyCFunction)_pymoose_ObjId_setField, METH_VARARGS,
         "Set specified attribute of the element."},
        {"getId", (PyCFunction)_pymoose_ObjId_getId, METH_VARARGS,
         "return integer representation of the id of the element. This will be"
         "an ObjId represented as a 3-tuple"},
        {"getFieldNames", (PyCFunction)_pymoose_ObjId_getFieldNames, METH_VARARGS,
         "Returns a tuple containing the field-names."
         "\n"
         "If one of 'valueFinfo', 'lookupFinfo', 'srcFinfo', 'destFinfo' or"
         "'sharedFinfo' is specified, then only fields of that type are"
         "returned. If no argument is passed, all fields are returned."},
        {"connect", (PyCFunction)_pymoose_ObjId_connect, METH_VARARGS,
         "Connect another object via a message."},
        {"getDataIndex", (PyCFunction)_pymoose_ObjId_getDataIndex, METH_VARARGS,
         "Get the index of this ObjId in the containing Id object."},
        {"getFieldIndex", (PyCFunction)_pymoose_ObjId_getFieldIndex, METH_VARARGS,
         "Get the index of this object as a field."},
        {NULL, NULL, 0, NULL},        /* Sentinel */        
    };
    /**
     * Method definitions.
     */    
    static PyMethodDef MooseMethods[] = {
        {"copy", (PyCFunction)_pymoose_copy, METH_VARARGS|METH_KEYWORDS, "Copy a Id object to a target."},
        {"move", (PyCFunction)_pymoose_move, METH_VARARGS, "Move a Id object to a destination."},
        {"useClock", (PyCFunction)_pymoose_useClock, METH_VARARGS, "Schedule objects on a specified clock"},
        {"setClock", (PyCFunction)_pymoose_setClock, METH_VARARGS, "Set the dt of a clock."},
        {"start", (PyCFunction)_pymoose_start, METH_VARARGS, "Start simulation"},
        {"reinit", (PyCFunction)_pymoose_reinit, METH_VARARGS, "Reinitialize simulation"},
        {"stop", (PyCFunction)_pymoose_stop, METH_VARARGS, "Stop simulation"},
        {"isRunning", (PyCFunction)_pymoose_isRunning, METH_VARARGS, "True if the simulation is currently running."},
        {"loadModel", (PyCFunction)_pymoose_loadModel, METH_VARARGS, "Load model from a file to a specified path."},
        {"getCwe", (PyCFunction)_pymoose_getCwe, METH_VARARGS, "Get the current working element."},
        {"setCwe", (PyCFunction)_pymoose_setCwe, METH_VARARGS, "Set the current working element."},
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };

    ///////////////////////////////////////////////
    // Type defs for PyObject of Id
    ///////////////////////////////////////////////
    static PyTypeObject IdType = { 
        PyObject_HEAD_INIT(0)               /* tp_head */
        0,                                  /* tp_internal */
        "moose.Id",                  /* tp_name */
        sizeof(_Id),                    /* tp_basicsize */
        0,                                  /* tp_itemsize */
        (destructor)_pymoose_Id_dealloc,                    /* tp_dealloc */
        0,                                  /* tp_print */
        0,                                  /* tp_getattr */
        0,                                  /* tp_setattr */
        0,                                  /* tp_compare */
        (reprfunc)_pymoose_Id_repr,                        /* tp_repr */
        0,                                  /* tp_as_number */
        &IdSequenceMethods,             /* tp_as_sequence */
        0,                                  /* tp_as_mapping */
        0,                                  /* tp_hash */
        0,                                  /* tp_call */
        (reprfunc)_pymoose_Id_str,               /* tp_str */
        PyObject_GenericGetAttr,            /* tp_getattro */
        PyObject_GenericSetAttr,            /* tp_setattro */
        0,                                  /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        "Id object of moose. Which can act as an array object.",
        0,                                  /* tp_traverse */
        0,                                  /* tp_clear */
        (richcmpfunc)_pymoose_Id_richCompare,       /* tp_richcompare */
        0,                                  /* tp_weaklistoffset */
        0,                                  /* tp_iter */
        0,                                  /* tp_iternext */
        IdMethods,                     /* tp_methods */
        0,                    /* tp_members */
        0,                                  /* tp_getset */
        0,                                  /* tp_base */
        0,                                  /* tp_dict */
        0,                                  /* tp_descr_get */
        0,                                  /* tp_descr_set */
        0,                                  /* tp_dictoffset */
        (initproc) _pymoose_Id_init,   /* tp_init */
        PyType_GenericAlloc,                /* tp_alloc */
        0,                  /* tp_new */
        0,                      /* tp_free */
    };

#define Id_Check(v) (Py_TYPE(v) == &IdType)
#define Id_SubtypeCheck(v) (PyType_IsSubtype(Py_TYPE(v),&IdType))

    ///////////////////////////////////////////////
    // Type defs for PyObject of ObjId
    ///////////////////////////////////////////////
    static PyTypeObject ObjIdType = { 
        PyObject_HEAD_INIT(0)               /* tp_head */
        0,                                  /* tp_internal */
        "moose.ObjId",                  /* tp_name */
        sizeof(_ObjId),                    /* tp_basicsize */
        0,                                  /* tp_itemsize */
        (destructor)_pymoose_ObjId_dealloc,                    /* tp_dealloc */
        0,                                  /* tp_print */
        0,                                  /* tp_getattr */
        0,                                  /* tp_setattr */
        0,                                  /* tp_compare */
        (reprfunc)_pymoose_ObjId_repr,                        /* tp_repr */
        0,                                  /* tp_as_number */
        0,                                  /* tp_as_sequence */
        0,                                  /* tp_as_mapping */
        0,                                  /* tp_hash */
        0,                                  /* tp_call */
        (reprfunc)_pymoose_ObjId_str,               /* tp_str */
        PyObject_GenericGetAttr,            /* tp_getattro */
        PyObject_GenericSetAttr,            /* tp_setattro */
        0,                                  /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        "Individual moose object contained in an array-type object.",
        0,                                  /* tp_traverse */
        0,                                  /* tp_clear */
        (richcmpfunc)_pymoose_ObjId_richCompare,       /* tp_richcompare */
        0,                                  /* tp_weaklistoffset */
        0,                                  /* tp_iter */
        0,                                  /* tp_iternext */
        ObjIdMethods,                     /* tp_methods */
        0,                    /* tp_members */
        0,                                  /* tp_getset */
        0,                                  /* tp_base */
        0,                                  /* tp_dict */
        0,                                  /* tp_descr_get */
        0,                                  /* tp_descr_set */
        0,                                  /* tp_dictoffset */
        (initproc) _pymoose_ObjId_init,   /* tp_init */
        PyType_GenericAlloc,                /* tp_alloc */
        0,                  /* tp_new */
        0,                      /* tp_free */
    };

#define ObjId_Check(v) (Py_TYPE(v) == &ObjIdType)
#define ObjId_SubtypeCheck(v) (Py_TYPE(v) == &ObjIdType)
    

    /* module initialization */
    PyMODINIT_FUNC init_moose()
    {
        PyObject *moose_module = Py_InitModule3("_moose", MooseMethods, "MOOSE = Multiscale Object-Oriented Simulation Environment.");
        if (moose_module == NULL)
            return;
        char moose_err[] = "moose.error";
        MooseError = PyErr_NewException(moose_err, NULL, NULL);
        Py_INCREF(MooseError);
        PyModule_AddObject(moose_module, "error", MooseError);
        IdType.ob_type = &PyType_Type;
        IdType.tp_new = PyType_GenericNew;
        IdType.tp_free = _PyObject_Del;
        if (PyType_Ready(&IdType) < 0)
            return;
        Py_INCREF(&IdType);
        PyModule_AddObject(moose_module, "Id", (PyObject*)&IdType);
        ObjIdType.ob_type = &PyType_Type;
        ObjIdType.tp_new = PyType_GenericNew;
        ObjIdType.tp_free = _PyObject_Del;
        if (PyType_Ready(&ObjIdType) < 0)
            return;
        Py_INCREF(&ObjIdType);
        PyModule_AddObject(moose_module, "ObjId", (PyObject*)&ObjIdType);
        
        setup_runtime_env(true);
        getShell();
        assert (Py_AtExit(&finalize) == 0);                
        PyModule_AddIntConstant(moose_module, "SINGLETHREADED", isSingleThreaded);
        PyModule_AddIntConstant(moose_module, "NUMCORES", numCores);
        PyModule_AddIntConstant(moose_module, "NUMNODES", numNodes);
        PyModule_AddIntConstant(moose_module, "MYNODE", myNode);
        PyModule_AddIntConstant(moose_module, "INFINITE", isInfinite);
        
    }
    
    //////////////////////////////////////////////////
    // Id functions
    //////////////////////////////////////////////////
    
    static int _pymoose_Id_init(_Id * self, PyObject * args, PyObject * kwds)
    {
        static char * kwlist[] = {"path", "dims", "type", NULL};
        char * path, * type = "Neutral";
        PyObject * dims = NULL;
        PyObject * src = NULL;
        // if (PyDict_Type.tp_init((PyObject *)self, args, kwds) < 0)
        //     return -1;
        
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|Os", kwlist, &path, &dims, &type)){
            if (!PyArg_ParseTuple(args, "O", &src)){
                return -1;
            }
            if (!Id_Check(src)){
                PyErr_SetString(PyExc_TypeError, "Id.__init__(path, dims, type) or Id.__init__(other_Id)");
                return -1;
            }
            _Id * src_cast = reinterpret_cast<_Id*>(src);
            if (!src_cast){
                PyErr_SetString(PyExc_TypeError, "Could not cast passed object to _Id.");
                return -1;
            }
            self->_id = src_cast->_id;
            return 0;
        }
        
        string trimmed_path(path);
        trimmed_path = trim(trimmed_path);
        size_t length = trimmed_path.length();
        if (length <= 0){
            PyErr_SetString(PyExc_ValueError, "path must be non-empty string.");
            return -1;
        }
        string trimmed_type = trim(string(type));
        if (trimmed_type.length() <= 0){
            PyErr_SetString(PyExc_ValueError, "type must be non-empty string.");
            return -1;
        }        

        //  paths ending with '/' should raise exception
        if ((length > 1) && (trimmed_path[length - 1] == '/')){
            PyErr_SetString(PyExc_ValueError, "Non-root path must not end with '/'");
            return -1;
        }
        vector <unsigned int> vec_dims;
        if (dims!= NULL && PySequence_Check(dims)){
            Py_ssize_t len = PySequence_Length(dims);
            for (Py_ssize_t ii = 0; ii < len; ++ ii){
                PyObject* dim = PySequence_GetItem(dims, ii);
                long dim_value = PyInt_AsLong(dim);
                if (dim_value == -1 && PyErr_Occurred()){
                    return -1;
                }
                vec_dims.push_back((unsigned int)dim_value);
            }                
        }
        if (vec_dims.empty()){
            vec_dims.push_back(1);
        }
        self->_id = Id(path);
        // If object does not exist, create new
        if ((self->_id == Id()) && (trimmed_path != "/") && (trimmed_path != "/root")){
            string parent_path;
            if (trimmed_path[0] != '/'){
                parent_path = getShell().getCwe().path();
            }
            size_t pos = trimmed_path.rfind("/");
            string name;
            if (pos != string::npos){
                name = trimmed_path.substr(pos+1);
                parent_path += "/";
                parent_path += trimmed_path.substr(0, pos+1);
            } else {
                name = trimmed_path;
            }
            self->_id = getShell().doCreate(string(type), Id(parent_path), string(name), vector<unsigned int>(vec_dims));
        } 
        return 0;            
    }// ! _pymoose_Id_init

    static void _pymoose_Id_dealloc(_Id * self)
    {
        PyObject_Del(self);
    } // ! _pymoose_Id_dealloc
    
    // 2011-03-23 15:14:11 (+0530)
    // 2011-03-26 17:02:19 (+0530)
    //
    // 2011-03-26 19:14:34 (+0530) - This IS UGLY! Destroying one
    // ObjId will destroy the containing element and invalidate all
    // the other ObjId with the same Id.
    // 2011-03-28 13:44:49 (+0530)
    static PyObject * _pymoose_Id_destroy(_Id * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":destroy")){
            return NULL;
        }
        self->_id.destroy();        
        Py_DECREF((PyObject*)self);
        Py_RETURN_NONE;
    }
    static PyObject * _pymoose_Id_repr(_Id * self)
    {
        return PyString_FromFormat("<Id: Id=%u>", self->_id.value());
    } // !  _pymoose_Id_repr
    static PyObject * _pymoose_Id_str(_Id * self)
    {
        return PyString_FromFormat("%s", Id::id2str(self->_id).c_str());
    } // !  _pymoose_Id_str

    // 2011-03-23 15:09:19 (+0530)
    static PyObject* _pymoose_Id_getId(_Id * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":Id.id")){
            return NULL;
        }
        unsigned int id = self->_id.value();        
        PyObject * ret = Py_BuildValue("I", id);
        return ret;
    }
    /**
       Not to be redone. 2011-03-23 14:42:48 (+0530)
     */
    static PyObject * _pymoose_Id_getPath(_Id * self, PyObject * args)
    {
        PyObject * obj = NULL;
        if (!PyArg_ParseTuple(args, ":Id.path", &obj)){
            return NULL;
        }
        string path = self->_id.path();
        PyObject * ret = Py_BuildValue("s", path.c_str());
        return ret;
    }
    static PyObject * _pymoose_Id_syncDataHandler(_Id * self, PyObject * args)
    {
        char * sizeField;
        PyObject * target;
        if(!PyArg_ParseTuple(args, "sO", &sizeField, &target)){
            return NULL;
        }
        _Id * tgt = reinterpret_cast<_Id*>(target);
        if (!tgt){
            PyErr_SetString(PyExc_TypeError, "Could not cast target to Id object");
            return NULL;
        }
        getShell().doSyncDataHandler(self->_id, string(sizeField), tgt->_id);
        Py_RETURN_NONE;        
    }

    /** Subset of sequence protocol functions */
    static Py_ssize_t _pymoose_Id_getLength(_Id * self)
    {
        vector< unsigned int> dims = Field< vector <unsigned int> >::get(ObjId(self->_id), "dimensions");
        if (dims.empty()){
            return (Py_ssize_t)1; // this is a bug in basecode - dimension 1 is returned as an empty vector
        } else {
            return (Py_ssize_t)dims[0];
        }
    }
    static PyObject * _pymoose_Id_getShape(_Id * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":getShape")){
            return NULL;
        }
        vector< unsigned int> dims = Field< vector <unsigned int> >::get(self->_id, "dimensions");
        if (dims.empty()){
            dims.push_back(1);
        }
        PyObject * ret = PyTuple_New((Py_ssize_t)dims.size());
        Py_XINCREF(ret);        
        for (unsigned int ii = 0; ii < dims.size(); ++ii){
            if (PyTuple_SetItem(ret, (Py_ssize_t)ii, Py_BuildValue("I", dims[ii]))){
                Py_XDECREF(ret);
                return NULL;
            }
        }
        return ret;
    }
    static PyObject * _pymoose_Id_getItem(_Id * self, Py_ssize_t index)
    {
        if (index < 0){
            index += _pymoose_Id_getLength(self);
        }
        if ((index < 0) || (index >= _pymoose_Id_getLength(self))){
            PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
            return NULL;
        }
        _ObjId * ret = PyObject_New(_ObjId, &ObjIdType);
        ret->_id = ObjId(self->_id, index);
        return (PyObject*)ret;
    }
    static PyObject * _pymoose_Id_getSlice(_Id * self, PyObject * args)
    {
        Py_ssize_t start, end;
        if (!PyArg_ParseTuple(args, "ii", &start, &end)){
            return NULL;
        }
        Py_ssize_t len = _pymoose_Id_getLength(self);
        while (start < 0){
            start += len;
        }
        while (end < 0){
            end += len;
        }
        if (start > end){
            PyErr_SetString(PyExc_IndexError, "Start index must be less than end.");
            return NULL;
        }
        PyObject * ret = PyTuple_New((Py_ssize_t)(end - start));
        Py_XINCREF(ret);        
        for (unsigned int ii = start; ii < end; ++ii){
            _ObjId * value = PyObject_New(_ObjId, &ObjIdType);
            value->_id = ObjId(self->_id, ii);
            if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)value)){
                Py_XDECREF(ret);
                return NULL;
            }
        }
        return ret;
    }
    
    static int _pymoose_Id_richCompare(_Id * self, PyObject * args, int op)
    {
        PyObject * other;
        if (!PyArg_ParseTuple(args, "O", &other)){
            return 0;
        }
        if (op == Py_EQ){
            return (self->_id == ((_Id*)other)->_id);
        } else if (op == Py_NE) {
            return (self->_id != ((_Id*)other)->_id);
        } else if (op == Py_LT){
            return (self->_id < ((_Id*)other)->_id);
        } else if (op == Py_GT) {
            return (((_Id*)other)->_id < self->_id);
        } else if (op == Py_LE){
            return ((self->_id < ((_Id*)other)->_id) || (self->_id == ((_Id*)other)->_id));
        } else if (op == Py_GE){
            return ((((_Id*)other)->_id < self->_id) || (self->_id == ((_Id*)other)->_id));
        } else {
            return 0;
        }
    }
    
    static int _pymoose_Id_contains(_Id * self, PyObject * args)
    {
        PyObject * obj = NULL;
        if (!PyArg_ParseTuple(args, "O:asSequence", &obj)){
            return 0;
        }
        if (!ObjId_Check(obj)){
            return 0;
        }
        return (((_ObjId*)obj)->_id.id == self->_id);
    }
    /////////////////////////////////////////////////////
    // ObjId functions.
    /////////////////////////////////////////////////////

    static int _pymoose_ObjId_init(_ObjId * self, PyObject * args, PyObject * kwargs)
    {
        unsigned int id = 0, data = 0, field = 0;
        PyObject * neutral;
        static char * kwlist[] = {"id", "dataIndex", "fieldIndex", NULL};
        if (PyArg_ParseTupleAndKeywords(args, kwargs, "I|II", kwlist, &id, &data, &field)){
            self->_id = ObjId(Id(id), DataId(data, field));
            return 0;
        } else if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|II", kwlist, &neutral, &data, &field)){
            if (!Id_Check(neutral)){
                PyErr_SetString(PyExc_TypeError, "ObjId.__init__(self, id, dataindex, fieldindex=0) or ObjId.__init__(self, Id, dataIndex, fieldIndex=0)");
                return -1;
            }
            self->_id = ObjId(((_Id*)neutral)->_id, DataId(data, field));
            return 0;
        } else {
            return -1;
        }        
    }
    
    static PyObject * _pymoose_ObjId_repr(_ObjId * self)
    {
        return PyString_FromFormat("<Id: Id=%u, Data=%u, Field=%u>", self->_id.id.value(), self->_id.dataId.data(), self->_id.dataId.field());
    } // !  _pymoose_ObjId_repr
    static PyObject * _pymoose_ObjId_str(_ObjId * self)
    {
        assert(self);
        return PyString_FromFormat("<Id: Id=%u, Data=%u, Field=%u>", self->_id.id.value(), self->_id.dataId.data(), self->_id.dataId.field());
    } // !  _pymoose_ObjId_str
    
    static void _pymoose_ObjId_dealloc(_ObjId * self)
    {
        PyObject_Del(self);
    } // ! _pymoose_ObjId_dealloc

    static PyObject* _pymoose_ObjId_getId(_ObjId * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":ObjId.getId")){
            return NULL;
        }
        unsigned int id = self->_id.id.value();
        PyObject * ret = Py_BuildValue("(I)", id);
        return ret;
    }


    static PyObject * _pymoose_ObjId_getFieldType(_ObjId * self, PyObject * args)
    {
        char * fieldName = NULL;
        if (!PyArg_ParseTuple(args, "s", &fieldName)){
            return NULL;
        }
        string typeStr = getFieldType(self->_id, string(fieldName));
        if (typeStr.length() <= 0){
            PyErr_SetString(PyExc_ValueError, "Empty string for field type. Field name may be incorrect.");
            return NULL;
        }
        PyObject * type = PyString_FromString(typeStr.c_str());
        return type;
    }  // ! _pymoose_Id_getFieldType

    /**
       2011-03-28 13:59:41 (+0530)
       
       Get a specified field. Re-done on: 2011-03-23 14:42:03 (+0530)

       I wonder how to cleanly do this. The Id - ObjId dichotomy is
       really ugly. When you don't pass an index, it is just treated
       as 0. Then what is the point of having Id separately? ObjId
       would been just fine!
     */
    static PyObject * _pymoose_ObjId_getField(_ObjId * self, PyObject * args)
    {
        const char * field = NULL;
        char ftype;
        if (!PyArg_ParseTuple(args, "s", &field)){
            return NULL;
        }
        PyObject * ret;
        // The GET_FIELD macro is just a short-cut to reduce typing
        // TYPE is the full type string for the field. TYPEC is the corresponding Python Py_BuildValue format character.
#define GET_FIELD(TYPE, TYPEC)                                          \
        { \
            TYPE value = Field<TYPE>::get(self->_id, string(field));    \
            ret = Py_BuildValue(#TYPEC, value);                         \
            break;                                                      \
        }                                                               \
        
#define GET_VECFIELD(TYPE, TYPEC) \
        {                                                               \
                vector<TYPE> val = Field< vector<TYPE> >::get(self->_id, string(field)); \
                ret = PyTuple_New((Py_ssize_t)val.size());              \
                for (unsigned int ii = 0; ii < val.size(); ++ ii ){     \
                        PyObject * entry = Py_BuildValue(#TYPEC, val[ii]); \
                        if (!entry || PyTuple_SetItem(ret, (Py_ssize_t)ii, entry)){ \
                            Py_XDECREF(ret);                             \
                            ret = NULL;                                 \
                            break;                                      \
                        }                                               \
                }                                                       \
                break;                                                  \
        }                                                               \
        
        string type = getFieldType(self->_id, string(field));
        if (type.empty()){
            string msg = "No such field on object ";
            msg += self->_id.id.path() + ": ";
            msg += field;
            PyErr_SetString(PyExc_AttributeError, msg.c_str());
            return NULL;
        }
        ftype = shortType(type);
        if (!ftype){
            string msg = "Type ";
            msg += type + " is not handled yet.";
            PyErr_SetString(PyExc_NotImplementedError, msg.c_str());
            return NULL;
        }
        switch(ftype){
            case 'c': GET_FIELD(char, c)
            case 'i': GET_FIELD(int, i)
            case 'h': GET_FIELD(short, h)
            case 'l': GET_FIELD(long, l)        
            case 'I': GET_FIELD(unsigned int, I)
            case 'k': GET_FIELD(unsigned long, k)
            case 'f': GET_FIELD(float, f)
            case 'd': GET_FIELD(double, d)
            case 's': {
                string _s = Field<string>::get(self->_id, string(field));
                ret = Py_BuildValue("s", _s.c_str());
                break;
            }
            case 'x':
                {                    
                    Id value = Field<Id>::get(self->_id, string(field));
                    PyObject * ret = (PyObject*)PyObject_New(_Id, &IdType);
                    ((_Id*)ret)->_id = value;
                    break;
                }
            case 'y':
                {
                    ObjId value = Field<ObjId>::get(self->_id, string(field));
                    PyObject * ret = (PyObject*)PyObject_New(_ObjId, &ObjIdType);
                    ((_ObjId*)ret)->_id = value;
                    break;
                }
            case 'z':
                {
                    PyErr_SetString(PyExc_NotImplementedError, "DataId handling not implemented in PyMoose yet.");
                    return NULL;
                }
            case 'v': GET_VECFIELD(int, i)
            case 'w': GET_VECFIELD(short, h)
            case 'L': GET_VECFIELD(long, l)        
            case 'U': GET_VECFIELD(unsigned int, I)        
            case 'K': GET_VECFIELD(unsigned long, k)        
            case 'F': GET_VECFIELD(float, f)        
            case 'D': GET_VECFIELD(double, d)        
            case 'S': {                                                 
                vector<string> val = Field< vector<string> >::get(self->_id, string(field)); 
                ret = PyTuple_New((Py_ssize_t)val.size());
                for (unsigned int ii = 0; ii < val.size(); ++ ii ){     
                    PyObject * entry = Py_BuildValue("s", val[ii].c_str()); 
                    if (!entry || PyTuple_SetItem(ret, (Py_ssize_t)ii, entry)){ 
                        Py_XDECREF(ret);                                  
                        ret = NULL;                                 
                        break;                                      
                    }                                               
                }                                                       
                break;                                                  
            }
            case 'X': // vector<Id>
                {
                    vector<Id> value = Field< vector <Id> >::get(self->_id, string(field));
                    PyObject * ret = PyTuple_New((Py_ssize_t)value.size());
                    for (unsigned int ii = 0; ii < value.size(); ++ii){
                        _Id * entry = PyObject_New(_Id, &IdType);
                        if (!entry){
                            Py_XDECREF(ret);
                            return NULL;
                        }
                        entry->_id = value[ii];
                        if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){
                            Py_XDECREF(ret);
                            return NULL;
                        }
                        cout << "Id: " << entry->_id << endl;
                    }
                    break;
                }
            case 'Y': // vector<ObjId>
                {
                    vector<ObjId> value = Field< vector <ObjId> >::get(self->_id, string(field));
                    PyObject * ret = PyTuple_New(value.size());
                    for (unsigned int ii = 0; ii < value.size(); ++ii){
                        _ObjId * entry = PyObject_New(_ObjId, &ObjIdType);                       
                        if (!entry){
                            Py_XDECREF(ret);
                            return NULL;
                        }
                        entry->_id = value[ii];
                        if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){
                            Py_XDECREF(ret);
                            return NULL;
                        }
                    }
                    break;
                }
                
            default:
                cout << type << " " << ftype << endl;
                PyErr_SetString(PyExc_TypeError, "Unrecognized field type.");
                ret = NULL;            
        }
#undef GET_FIELD    
#undef GET_VECFIELD        
        return ret;        
    }
    /**
       Set a specified field. Redone on 2011-03-23 14:41:45 (+0530)
     */
    static PyObject * _pymoose_ObjId_setField(_ObjId * self, PyObject * args)
    {
        
#define SET_FIELD(TYPE, TYPEC)                          \
        {                                               \
            TYPE _value;                                        \
            if (!PyArg_ParseTuple(value, #TYPEC, &_value)){     \
                return NULL;                                    \
            }                                                           \
            ret = Field<TYPE>::set(self->_id, string(field), _value);   \
            break;                                                      \
        } //! SET_FIELD

        // SET_VECFIELD(TYPE, TYPEC) -- macro to set a vector field.
        // TYPE - C/C++ basic data type of the vector elements,
        // TYPEC - Python typecode passed to PyArg_ParseTuple.
#define SET_VECFIELD(TYPE, TYPEC)                                       \
        {                                                               \
            if (!PySequence_Check(value)){                              \
                PyErr_SetString(                                        \
                        PyExc_TypeError,                                \
                        "For setting vector<int> field, specified value must be a sequence." ); \
                return NULL;                                            \
            }                                                           \
            vector<TYPE> _value;                                        \
            TYPE entry;                                                 \
            Py_ssize_t length = PySequence_Length(value);               \
            for (unsigned int ii = 0; ii < length; ++ii){               \
                if (!PyArg_ParseTuple(PySequence_GetItem(value, ii), #TYPEC, &entry)){ \
                    return NULL;                                        \
                }                                                       \
                _value.push_back(entry);                                \
            }                                                           \
            ret = Field< vector <TYPE> >::set(self->_id, string(field), _value); \
            break;                                                          \
        }
        PyObject * value;
        char * field;
        int ret = 0;
        if (!PyArg_ParseTuple(args, "sO", &field,  &value)){
            cout << "Here" << endl;
            return NULL;
        }
        char ftype = shortType(getFieldType(self->_id, string(field)));
        
        if (!ftype){
            PyErr_SetString(PyExc_AttributeError, "Field not valid.");
        }
        
        switch(ftype){
            case 'c': SET_FIELD(char, c)
            case 'i': SET_FIELD(int, i)
            case 'h': SET_FIELD(int, h)
            case 'l': SET_FIELD(long, l)
            case 'I': SET_FIELD(unsigned int, I)
            case 'k': SET_FIELD(unsigned long, k)
            case 'f': SET_FIELD(float, f)
            case 'd': SET_FIELD(double, d)
            case 's': 
                {
                    char * _value = PyString_AsString(value);
                    if (_value){
                        ret = Field<string>::set(self->_id, string(field), string(_value));
                    } else {
                        return NULL;
                    }
                    break;
                }
            case 'x': // Id
                {
                    if (value){
                        ret = Field<Id>::set(self->_id, string(field), ((_Id*)value)->_id);
                    } else {
                        PyErr_SetString(PyExc_ValueError, "Null pointer passed as Id value.");
                        return NULL;
                    }
                    break;
                }
            case 'y': // ObjId
                {
                    if (value){
                        ret = Field<ObjId>::set(self->_id, string(field), ((_ObjId*)value)->_id);
                    } else {
                        PyErr_SetString(PyExc_ValueError, "Null pointer passed as Id value.");
                        return NULL;
                    }
                    break;
                }
            case 'z': // DataId
                {
                    PyErr_SetString(PyExc_NotImplementedError, "DataId handling not implemented in PyMoose yet.");
                    return NULL;
                }
            case 'v': SET_VECFIELD(int, i)
            case 'w': SET_VECFIELD(short, h)
            case 'L': SET_VECFIELD(long, l)
            case 'U': SET_VECFIELD(unsigned int, I)
            case 'K': SET_VECFIELD(unsigned long, k)
            case 'F': SET_VECFIELD(float, f)
            case 'D': SET_VECFIELD(double, d)
            case 'S':
                {
                    if (!PySequence_Check(value)){
                        PyErr_SetString(PyExc_TypeError, "For setting vector<string> field, specified value must be a sequence." );
                        return NULL;
                    }
                    Py_ssize_t length = PySequence_Length(value);
                    vector<string> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        char * v = PyString_AsString(PySequence_GetItem(value, ii));
                        _value.push_back(string(v));
                    }
                    ret = Field< vector < string > >::set(self->_id, string(field), _value);
                    break;
                }
                
            default:
                ret = 0;
                break;
        }
        
        if (ret){
            Py_RETURN_NONE;
        } else {
            PyErr_SetString(PyExc_TypeError, "The specified field type is not valid.");
            return NULL;
        }
#undef SET_FIELD
#undef SET_VECFIELD        
    } // _pymoose_Id_setField
    
    // 2011-03-23 15:28:26 (+0530)
    static PyObject * _pymoose_ObjId_getFieldNames(_ObjId * self, PyObject *args)
    {
        char * ftype = NULL;
        if (!PyArg_ParseTuple(args, "|s", &ftype)){
            return NULL;
        }else if ( ftype && (strlen(ftype) == 0)){
            PyErr_SetString(PyExc_ValueError, "Field type must be a non-empty string");
            return NULL;
        }
        string ftype_str = (ftype != NULL)? string(ftype): "";
        vector<string> ret;
        if (ftype_str == ""){
            static char * fieldTypes[] = {"valueFinfo", "srcFinfo", "destFinfo", "lookupFinfo", "sharedFinfo", 0};
            char ** a;
            for (a = &fieldTypes[0]; *a; ++a){
                vector<string> fields = getFieldNames(self->_id, string(*a));
                ret.insert(ret.end(), fields.begin(), fields.end());
            }            
        } else {
            ret = getFieldNames(self->_id, ftype_str);
        }
        cout << "getFieldNames: ret.size() = " << ret.size() << endl;
        
        PyObject * pyret = PyTuple_New((Py_ssize_t)ret.size());
        for (unsigned int ii = 0; ii < ret.size(); ++ ii ){
            PyObject * fname = Py_BuildValue("s", ret[ii].c_str());
            if (!fname){
                Py_XDECREF(pyret);
                pyret = NULL;
                break;
            }
            if (PyTuple_SetItem(pyret, (Py_ssize_t)ii, fname)){
                Py_XDECREF(pyret);
                pyret = NULL;
                break;
            }
        }
        return pyret;             
    }
    
    // 2011-03-28 10:10:19 (+0530)
    // 2011-03-23 15:13:29 (+0530)
    // getChildren is not required as it can be accessed as getField("children")

    // 2011-03-28 10:51:52 (+0530)
    static PyObject * _pymoose_ObjId_connect(_ObjId * self, PyObject * args)
    {
        PyObject * destPtr = NULL;
        char * srcField, * destField, * msgType;

        if(!PyArg_ParseTuple(args, "sOss", &srcField, &destPtr, &destField, &msgType)){
            return NULL;
        }
        _ObjId * dest = reinterpret_cast<_ObjId*>(destPtr);
        bool ret = (getShell().doAddMsg(msgType, self->_id, string(srcField), dest->_id, string(destField)) != Msg::badMsg);
        if (!ret){
            PyErr_SetString(PyExc_NameError, "connect failed: check field names and type compatibility.");
            return NULL;
        }
        return Py_BuildValue("i", ret);
    }

    static int _pymoose_ObjId_richCompare(_ObjId * self, PyObject * other, int op)
    {
        if (op == Py_EQ){
            return (self->_id == ((_ObjId*)other)->_id);
        } else if (op == Py_NE){
            return !(self->_id == ((_ObjId*)other)->_id);
        } else {
            return 0;
        }
    }

    static PyObject * _pymoose_ObjId_getDataIndex(_ObjId * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":getDataIndex")){
            return NULL;
        }
        PyObject * ret = Py_BuildValue("I", self->_id.dataId.data());
        return ret;
    }
    static PyObject * _pymoose_ObjId_getFieldIndex(_ObjId * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":getFieldIndex")){
            return NULL;
        }
        PyObject * ret = Py_BuildValue("I", self->_id.dataId.field());
        return ret;
    } 
    ////////////////////////////////////////////
    // The following are global functions
    ////////////////////////////////////////////

    
    static PyObject * _pymoose_copy(PyObject * dummy, PyObject * args, PyObject * kwargs)
    {
        PyObject * src, *dest;
        char * newName;
        char * kwlist[] = {"src", "dest", "name", "n", "copyMsg", NULL};
        unsigned int num=1, copyExtMsgs=1;
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOs|II", kwlist, &src, &dest, &newName, &num, &copyExtMsgs)){
            return NULL;
        }
        if (!Id_SubtypeCheck(src)){
            PyErr_SetString(PyExc_TypeError, "Source must be instance of Id.");
            return NULL;
        } else if (!Id_SubtypeCheck(dest)){
            PyErr_SetString(PyExc_TypeError, "Destination must be instance of Id.");
            return NULL;
        }
        
        _Id * tgt = PyObject_New(_Id, &IdType);
        
        tgt->_id = getShell().doCopy(((_Id*)src)->_id, ((_Id*)dest)->_id, string(newName), num, copyExtMsgs);
        PyObject * ret = (PyObject*)tgt;
        return ret;            
    }

// Not sure what this function should return... ideally the Id of the
// moved object - does it change though?
    static PyObject * _pymoose_move(PyObject * dummy, PyObject * args)
    {
        PyObject * src, * dest;
        if (!PyArg_ParseTuple(args, "OO", &src, &dest)){
            return NULL;
        }
        getShell().doMove(((_Id*)src)->_id, ((_Id*)dest)->_id);
        Py_RETURN_NONE;
    }


    static PyObject * _pymoose_useClock(PyObject * dummy, PyObject * args)
    {
        char * path, * field;
        unsigned int tick;
        if(!PyArg_ParseTuple(args, "Iss", &tick, &path, &field)){
            return NULL;
        }
        getShell().doUseClock(string(path), string(field), tick);
        Py_RETURN_NONE;
    }
    static PyObject * _pymoose_setClock(PyObject * dummy, PyObject * args)
    {
        unsigned int tick;
        double dt;
        if(!PyArg_ParseTuple(args, "Id", &tick, &dt)){
            return NULL;
        }
        if (dt < 0){
            PyErr_SetString(PyExc_ValueError, "dt must be positive.");
            return NULL;
        }
        getShell().doSetClock(tick, dt);
        Py_RETURN_NONE;
    }
    static PyObject * _pymoose_start(PyObject * dummy, PyObject * args)
    {
        double runtime;
        if(!PyArg_ParseTuple(args, "d", &runtime)){
            return NULL;
        }
        if (runtime <= 0.0){
            PyErr_SetString(PyExc_ValueError, "simulation runtime must be positive.");
            return NULL;
        }
        getShell().doStart(runtime);
        Py_RETURN_NONE;
    }
    static PyObject * _pymoose_reinit(PyObject * dummy, PyObject * args)
    {
        getShell().doReinit();
        Py_RETURN_NONE;
    }
    static PyObject * _pymoose_stop(PyObject * dummy, PyObject * args)
    {
        getShell().doStop();
        Py_RETURN_NONE;
    }
    static PyObject * _pymoose_isRunning(PyObject * dummy, PyObject * args)
    {
        return Py_BuildValue("i", getShell().isRunning());
    }
    
    static PyObject * _pymoose_loadModel(PyObject * dummy, PyObject * args)
    {
        char * fname, * modelpath;
        if(!PyArg_ParseTuple(args, "ss", &fname, &modelpath)){
            return NULL;
        }
        
        _Id * model = new _Id();
        model->_id = getShell().doLoadModel(string(fname), string(modelpath));
        PyObject * ret = reinterpret_cast<PyObject*>(model);
        return ret;
    }

    static PyObject * _pymoose_setCwe(PyObject * dummy, PyObject * args)
    {
        PyObject * element = NULL;
        char * path = "/";
        Id id;
        if(PyArg_ParseTuple(args, "s", &path)){
            id = Id(string(path));
        } else if (PyArg_ParseTuple(args, "O", &element)){
            id = (reinterpret_cast<_Id*>(element))->_id;
        } else {
            return NULL;
        }
        getShell().setCwe(id);
        Py_RETURN_NONE;
    }

    static PyObject * _pymoose_getCwe(PyObject * dummy, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":getCwe")){
            return NULL;
        }
        _Id * cwe = (_Id*)PyObject_New(_Id, &IdType);
        cwe->_id = getShell().getCwe();        
        PyObject * ret = (PyObject*)cwe;
        return ret;
    }

    
    
    
} // end extern "C"



//////////////////////////////////////////////
// Main function
//////////////////////////////////////////////


int main(int argc, char* argv[])
{
    for (int ii = 0; ii < argc; ++ii){
        cout << "ARGV: " << argv[ii];
    }
    cout << endl;
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    init_moose();
    return 0;
}

// 
// moosemodule.cpp ends here
