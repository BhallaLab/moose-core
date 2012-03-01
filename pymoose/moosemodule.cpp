// moosemodule.cpp --- 
// 
// Filename: moosemodule.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Thu Mar 10 11:26:00 2011 (+0530)
// Version: 
// Last-Updated: Fri Mar  2 01:03:01 2012 (+0530)
//           By: Subhasis Ray
//     Update #: 4923
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
// 2012-01-05 Much polished version. Handling destFinfos as methods in
// Python class.

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
#include "../randnum/randnum.h"
#include "../shell/Shell.h"

#include "pymoose.h"
#include "moosemodule.h"

using namespace std;
using namespace pymoose;

extern void nonMpiTests(Shell *);
extern void mpiTests();
extern void processTests(Shell *);
extern void regressionTests();

extern int isSingleThreaded;
extern int isInfinite;
extern int numNodes;
extern int numCores;
extern int myNode;
extern const char ** FINFOTYPES;

// 
// C wrappers for C++ classes
// This is used by Python
extern "C" {
    static PyMethodDef IdMethods[] = {
        // {"init", (PyCFunction)_pymoose_Id_init, METH_VARARGS,
        //  "Initialize a Id object."},
        {"delete", (PyCFunction)_pymoose_Id_delete, METH_VARARGS,
         "Delete the underlying moose element"},
        {"getValue", (PyCFunction)_pymoose_Id_getValue, METH_VARARGS,
         "Return integer representation of the id of the element."},
        {"getPath", (PyCFunction)_pymoose_Id_getPath, METH_VARARGS,
         "Return the path of this Id object."},
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

    static PyMethodDef ObjIdMethods[] = {
        {"getFieldType", (PyCFunction)_pymoose_ObjId_getFieldType, METH_VARARGS,
         "Get the string representation of the type of this field."},        
        {"getField", (PyCFunction)_pymoose_ObjId_getField, METH_VARARGS,
         "Get specified attribute of the element."},
        {"setField", (PyCFunction)_pymoose_ObjId_setField, METH_VARARGS,
         "Set specified attribute of the element."},
        {"getLookupField", (PyCFunction)_pymoose_ObjId_getLookupField, METH_VARARGS,
         "Lookup a field based on key."},
        {"setLookupField", (PyCFunction)_pymoose_ObjId_setLookupField, METH_VARARGS,
         "Set a lookup field value based on key."},
        {"getId", (PyCFunction)_pymoose_ObjId_getId, METH_VARARGS,
         "Return integer representation of the id of the element. This will be"
         "an ObjId represented as a 3-tuple"},
        {"getFieldNames", (PyCFunction)_pymoose_ObjId_getFieldNames, METH_VARARGS,
         "Return a tuple containing the field-names."
         "\n"
         "If one of 'valueFinfo', 'lookupFinfo', 'srcFinfo', 'destFinfo' or"
         "'sharedFinfo' is specified, then only fields of that type are"
         "returned. If no argument is passed, all fields are returned."},
        {"getNeighbors", (PyCFunction)_pymoose_ObjId_getNeighbors, METH_VARARGS,
         "Retrieve a list of Ids connected via this field."},
        {"connect", (PyCFunction)_pymoose_ObjId_connect, METH_VARARGS,
         "Connect another object via a message."},
        {"getDataIndex", (PyCFunction)_pymoose_ObjId_getDataIndex, METH_VARARGS,
         "Get the index of this ObjId in the containing Id object."},
        {"getFieldIndex", (PyCFunction)_pymoose_ObjId_getFieldIndex, METH_VARARGS,
         "Get the index of this object as a field."},
        {"setDestField", (PyCFunction)_pymoose_ObjId_setDestField, METH_VARARGS,
         "Set a function field (DestFinfo). This should not be accessed directly. A python"
         " member method should be wrapping it for each DestFinfo in each MOOSE"
         " class. When used directly, it takes the form:\n"
         " {ObjId}.setDestField({destFinfoName}, {arg1},{arg2}, ... , {argN})\n"
         " where destFinfoName is the string representing the name of the"
         " DestFinfo refering to the target function, arg1, ..., argN are the"
         " arguments to be passed to the target function."
         " Return True on success, False on failure."},
        {NULL, NULL, 0, NULL},        /* Sentinel */        
    };
    /**
     * Method definitions.
     */    
    static PyMethodDef MooseMethods[] = {
        {"copy", (PyCFunction)_pymoose_copy, METH_VARARGS|METH_KEYWORDS, "Copy a Id object to a target."},
        {"move", (PyCFunction)_pymoose_move, METH_VARARGS, "Move a Id object to a destination."},
        {"delete", (PyCFunction)_pymoose_delete, METH_VARARGS, "Delete the moose object."},
        {"useClock", (PyCFunction)_pymoose_useClock, METH_VARARGS, "Schedule objects on a specified clock"},
        {"setClock", (PyCFunction)_pymoose_setClock, METH_VARARGS, "Set the dt of a clock."},
        {"start", (PyCFunction)_pymoose_start, METH_VARARGS, "Start simulation"},
        {"reinit", (PyCFunction)_pymoose_reinit, METH_VARARGS, "Reinitialize simulation"},
        {"stop", (PyCFunction)_pymoose_stop, METH_VARARGS, "Stop simulation"},
        {"isRunning", (PyCFunction)_pymoose_isRunning, METH_VARARGS, "True if the simulation is currently running."},
        {"exists", (PyCFunction)_pymoose_exists, METH_VARARGS, "True if there is an object with specified path."},
        {"loadModel", (PyCFunction)_pymoose_loadModel, METH_VARARGS, "Load model from a file to a specified path.\n"
         "Parameters:\n"
         "\tstr filename -- model description file.\n"
         "\tstr modelpath -- moose path for the top level element of the model to be created.\n"
         "\tstr solverclass -- (optional) solver type to be used for simulating the model.\n"},
        {"connect", (PyCFunction)_pymoose_connect, METH_VARARGS, "Create a message between srcField on src element to destField on target element."},        
        {"getCwe", (PyCFunction)_pymoose_getCwe, METH_VARARGS, "Get the current working element. 'pwe' is an alias of this function."},
        {"pwe", (PyCFunction)_pymoose_getCwe, METH_VARARGS, "Get the current working element. 'getCwe' is an alias of this function."},
        {"setCwe", (PyCFunction)_pymoose_setCwe, METH_VARARGS, "Set the current working element. 'ce' is an alias of this function"},
        {"ce", (PyCFunction)_pymoose_setCwe, METH_VARARGS, "Set the current working element. setCwe is an alias of this function."},
        {"getFieldDict", (PyCFunction)_pymoose_getFieldDict, METH_VARARGS, "Get dictionary of field names and types for specified class.\n"
         " Parameters:\n"
         "str className -- MOOSE class to find the fields of.\n"
         "str finfoType -- (optional) Finfo type of the fields to find. If empty or not specified, all fields will be retrieved."
        },
        {"syncDataHandler", (PyCFunction)_pymoose_syncDataHandler, METH_VARARGS,
         "synchronizes fieldDimension on the DataHandler"
         " across nodes. Used after function calls that might alter the"
         " number of Field entries in the table."
         " The target is the FieldElement whose fieldDimension needs updating."},
        {"seed", (PyCFunction)_pymoose_seed, METH_VARARGS, "Seed the random number generator of MOOSE."},
        {"wildcardFind", (PyCFunction)_pymoose_wildcardFind, METH_VARARGS, "Return a list of Ids by a wildcard query."},

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
        (hashfunc)_pymoose_Id_hash,                                  /* tp_hash */
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
        (hashfunc)_pymoose_ObjId_hash,         /* tp_hash */
        0,                                  /* tp_call */
        (reprfunc)_pymoose_ObjId_repr,               /* tp_str */
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
        PyModule_AddStringConstant(moose_module, "__version__", getShell().doVersion().c_str());
        PyModule_AddStringConstant(moose_module, "VERSION", getShell().doVersion().c_str());
        PyModule_AddStringConstant(moose_module, "SVN_REVISION", getShell().doRevision().c_str());
        
    }
    
    //////////////////////////////////////////////////
    // Id functions
    //////////////////////////////////////////////////
    
    static int _pymoose_Id_init(_Id * self, PyObject * args, PyObject * kwds)
    {
        static const char * kwlist[] = {"path", "dims", "type", NULL};
        char * path;
        const char * type = "Neutral";
        PyObject * dims = NULL;
        PyObject * src = NULL;
        unsigned int id = 0;
        if (PyArg_ParseTuple(args, "I:_pymoose_Id_init", &id)){
            self->id_ = Id(id);
            return 0;
        }
        
        if (PyArg_ParseTuple(args, "O:_pymoose_Id_init", &src) && Id_Check(src)){
            self->id_ = ((_Id*)src)->id_;
            return 0;
        }
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|Os:_pymoose_Id_init", const_cast<char**>(kwlist), &path, &dims, &type)){
            PyErr_SetString(PyExc_TypeError, "Invalid paramaters. Id.__init__ has the following signature: "
                            "Id.__init__(path, dims, type) or Id.__init__(other_Id) or "
                            "Id.__init__(id_value)");
            return -1;
        }

        PyErr_Clear();

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
        vector <int> vec_dims;
        Py_ssize_t len = 1;
        if (dims){
            if (PySequence_Check(dims)){
                len = PySequence_Length(dims);
                for (Py_ssize_t ii = 0; ii < len; ++ ii){
                    PyObject* dim = PySequence_GetItem(dims, ii);
                    long dim_value = PyInt_AsLong(dim);
                    if ((dim_value == -1) && PyErr_Occurred()){
                        return -1;
                    }
                    vec_dims.push_back((unsigned int)dim_value);
                }
            } else if (PyInt_Check(dims)){
                len = PyInt_AsLong(dims);
                if (len <= 0){
                    len = 1;
                }
            }
        }
        
        if (vec_dims.empty()){
            vec_dims.push_back(len);
        }
        self->id_ = Id(path);
        // If object does not exist, create new
        if ((self->id_ == Id()) && (trimmed_path != "/") && (trimmed_path != "/root")){
            string parent_path;
            string name;
            size_t pos = trimmed_path.rfind("/");
            if (pos != string::npos){
                name = trimmed_path.substr(pos+1);
                parent_path = trimmed_path.substr(0, pos);
            } else {
                name = trimmed_path;
            }
            if (trimmed_path[0] != '/'){
                parent_path = getShell().getCwe().path() + parent_path;
            } else if (parent_path.empty()){
                parent_path = "/";
            }
                
            Id parent_id(parent_path);
            if (parent_id == Id() && parent_path != "/" && parent_path != "/root") {
                string message = "Parent element does not exist: ";
                message += parent_path;
                PyErr_SetString(PyExc_ValueError, message.c_str());
                return -1;
            }
            self->id_ = getShell().doCreate(string(type), parent_id, string(name), vector<int>(vec_dims));
        } 
        return 0;            
    }// ! _pymoose_Id_init

    static long _pymoose_Id_hash(_Id * self, PyObject * args)
    {
        return self->id_.value(); // hash is the same as the Id value
    }

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
    static PyObject * _pymoose_Id_delete(_Id * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":_pymoose_Id_delete")){
            return NULL;
        }
        if (self->id_ == Id()){
            PyErr_SetString(PyExc_ValueError, "Cannot delete moose shell.");
            return NULL;
        }
        getShell().doDelete(self->id_);
        self->id_ = Id();
        Py_RETURN_NONE;
    }
    static PyObject * _pymoose_Id_repr(_Id * self)
    {
        return PyString_FromFormat("<Id: id=%u, path=%s>", self->id_.value(), self->id_.path().c_str());
    } // !  _pymoose_Id_repr
    static PyObject * _pymoose_Id_str(_Id * self)
    {
        return PyString_FromFormat("%s", Id::id2str(self->id_).c_str());
    } // !  _pymoose_Id_str

    // 2011-03-23 15:09:19 (+0530)
    static PyObject* _pymoose_Id_getValue(_Id * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":_pymoose_Id_getValue")){
            return NULL;
        }
        unsigned int id = self->id_.value();        
        PyObject * ret = Py_BuildValue("I", id);
        return ret;
    }
    /**
       Not to be redone. 2011-03-23 14:42:48 (+0530)
    */
    static PyObject * _pymoose_Id_getPath(_Id * self, PyObject * args)
    {
        PyObject * obj = NULL;
        if (!PyArg_ParseTuple(args, ":_pymoose_Id_getPath", &obj)){
            return NULL;
        }
        string path = self->id_.path();
        PyObject * ret = Py_BuildValue("s", path.c_str());
        return ret;
    }

    /** Subset of sequence protocol functions */
    static Py_ssize_t _pymoose_Id_getLength(_Id * self)
    {
        vector< unsigned int> dims = Field< vector <unsigned int> >::get(ObjId(self->id_), "dimensions");
        if (dims.empty()){
            return (Py_ssize_t)1; // this is a bug in basecode - dimension 1 is returned as an empty vector
        } else {
            return (Py_ssize_t)dims[0];
        }
    }
    static PyObject * _pymoose_Id_getShape(_Id * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":_pymoose_Id_getShape")){
            return NULL;
        }
        vector< unsigned int> dims = Field< vector <unsigned int> >::get(self->id_, "dimensions");
        if (dims.empty()){
            dims.push_back(1);
        }
        PyObject * ret = PyTuple_New((Py_ssize_t)dims.size());
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
        ret->oid_ = ObjId(self->id_, index);
        return (PyObject*)ret;
    }
    static PyObject * _pymoose_Id_getSlice(_Id * self, PyObject * args)
    {
        Py_ssize_t start, end;
        if (!PyArg_ParseTuple(args, "ii:_pymoose_Id_getSlice", &start, &end)){
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
        // Py_XINCREF(ret);        
        for (unsigned int ii = start; ii < end; ++ii){
            _ObjId * value = PyObject_New(_ObjId, &ObjIdType);
            value->oid_ = ObjId(self->id_, ii);
            if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)value)){
                Py_XDECREF(ret);
                return NULL;
            }
        }
        return ret;
    }
    
    static PyObject * _pymoose_Id_richCompare(_Id * self, PyObject * other, int op)
    {
        int ret = 0;
        if (!self || !other){
            ret = 0;
        } else if (!Id_SubtypeCheck(other)){
            ret = 0;
        } else if (op == Py_EQ){
            ret = (self->id_ == ((_Id*)other)->id_);
        } else if (op == Py_NE) {
            ret = (self->id_ != ((_Id*)other)->id_);
        } else if (op == Py_LT){
            ret = (self->id_ < ((_Id*)other)->id_);
        } else if (op == Py_GT) {
            ret = (((_Id*)other)->id_ < self->id_);
        } else if (op == Py_LE){
            ret = ((self->id_ < ((_Id*)other)->id_) || (self->id_ == ((_Id*)other)->id_));
        } else if (op == Py_GE){
            ret = ((((_Id*)other)->id_ < self->id_) || (self->id_ == ((_Id*)other)->id_));
        } else {
            ret = 0;
        }
        return Py_BuildValue("i", ret);
    }
    
    static int _pymoose_Id_contains(_Id * self, PyObject * obj)
    {
        int ret = 0;
        if (ObjId_Check(obj)){
            ret = (((_ObjId*)obj)->oid_.id == self->id_);
        }
        return ret;
    }
    /////////////////////////////////////////////////////
    // ObjId functions.
    /////////////////////////////////////////////////////

    static int _pymoose_ObjId_init(_ObjId * self, PyObject * args, PyObject * kwargs)
    {
        unsigned int id = 0, data = 0, field = 0, numFieldBits = 0;
        PyObject * obj;
        static const char * kwlist[] = {"id", "dataIndex", "fieldIndex", "numFieldBits", NULL};
        if (PyArg_ParseTupleAndKeywords(args, kwargs, "I|III:_pymoose_ObjId_init", const_cast<char**>(kwlist), &id, &data, &field, &numFieldBits)){
            self->oid_ = ObjId(Id(id), DataId(data, field, numFieldBits));
            return 0;
        } else if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|III:_pymoose_ObjId_init", const_cast<char**>(kwlist), &obj, &data, &field, &numFieldBits)){
            PyErr_Clear();
            if (Id_Check(obj)){
                self->oid_ = ObjId(((_Id*)obj)->id_, DataId(data, field, numFieldBits));
                return 0;
            } else if (ObjId_Check(obj)){
                self->oid_ = ((_ObjId*)obj)->oid_;
                return 0;
            } else if (PyString_Check(obj)){
                string path(PyString_AsString(obj));
                self->oid_ = ObjId(path);
                if (ObjId::bad == self->oid_){
                    PyErr_SetString(PyExc_ValueError, "Path does not match any existing object. Give an arrayelement compatible path to create a new object.");
                    return -1;
                }
                return 0;
            } else {
                PyErr_SetString(PyExc_TypeError, "ObjId.__init__(self, id, dataindex, fieldindex=0, numFieldBits=0) or ObjId.__init__(self, Id, dataIndex, fieldIndex=0, numFieldBits=0) or ObjId.__init__(self, ObjId) or ObjId.__init__(self, path)");
                return -1;
            }            
        } else {
            PyErr_SetString(PyExc_TypeError, "Unrecognized parameters.");
            return -1;
        }        
    }

    /**
       This function simple returns the python hash of the unique path
       of this object.
    */
    static long _pymoose_ObjId_hash(_ObjId * self, PyObject * args)
    {
        PyObject * path = Py_BuildValue("s", self->oid_.path().c_str());        
        long ret = PyObject_Hash(path);
        Py_XDECREF(path);
        return ret;
    }
    
    static PyObject * _pymoose_ObjId_repr(_ObjId * self)
    {
        return PyString_FromFormat("<ObjId: id=%u, dataId=%lld, path=%s>", self->oid_.id.value(), self->oid_.dataId.value(), self->oid_.path().c_str());
    } // !  _pymoose_ObjId_repr

    
    static void _pymoose_ObjId_dealloc(_ObjId * self)
    {
        PyObject_Del(self);
    } // ! _pymoose_ObjId_dealloc

    static PyObject* _pymoose_ObjId_getId(_ObjId * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":ObjId.getId")){
            return NULL;
        }
        _Id * ret = PyObject_New(_Id, &IdType);
        ret->id_ = self->oid_.id;
        return (PyObject*)ret;
    }


    static PyObject * _pymoose_ObjId_getFieldType(_ObjId * self, PyObject * args)
    {
        char * fieldName = NULL;
        char * finfoType = NULL;
        if (!PyArg_ParseTuple(args, "s|s:_pymoose_ObjId_getFieldType", &fieldName, &finfoType)){
            return NULL;
        }
        string finfoTypeStr = "";
        if (finfoType != NULL){
            finfoTypeStr = finfoType;
        } else {
            finfoTypeStr = "valueFinfo";
        }
        string typeStr = getFieldType(self->oid_, string(fieldName), finfoTypeStr);
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
        if (!PyArg_ParseTuple(args, "s:_pymoose_ObjId_getField", &field)){
            return NULL;
        }
        PyObject * ret;
        // The GET_FIELD macro is just a short-cut to reduce typing
        // TYPE is the full type string for the field. TYPEC is the corresponding Python Py_BuildValue format character.
#define GET_FIELD(TYPE, TYPEC)                                          \
        {                                                               \
            TYPE value = Field<TYPE>::get(self->oid_, string(field));   \
            ret = Py_BuildValue(#TYPEC, value);                         \
            break;                                                      \
        }                                                               \
        
#define GET_VECFIELD(TYPE, TYPEC)                                       \
        {                                                               \
            vector<TYPE> val = Field< vector<TYPE> >::get(self->oid_, string(field)); \
            ret = PyTuple_New((Py_ssize_t)val.size());                  \
            for (unsigned int ii = 0; ii < val.size(); ++ ii ){         \
                PyObject * entry = Py_BuildValue(#TYPEC, val[ii]);      \
                if (!entry || PyTuple_SetItem(ret, (Py_ssize_t)ii, entry)){ \
                    Py_XDECREF(ret);                                    \
                    ret = NULL;                                         \
                    break;                                              \
                }                                                       \
            }                                                           \
            break;                                                      \
        }                                                               \

        
        string type = getFieldType(self->oid_, string(field), "valueFinfo");
        if (type.empty()){
            string msg = "No such field on object ";
            msg += self->oid_.id.path() + ": ";
            msg += field;
            PyErr_SetString(PyExc_AttributeError, msg.c_str());
            return NULL;
        }
        ftype = shortType(type);
        if (!ftype){
            string msg = "No short type for field ";
            msg += field;
            msg += " of type ";
            msg += type;
            PyErr_SetString(PyExc_NotImplementedError, msg.c_str());
            return NULL;
        }
        switch(ftype){
            case 'b': {
                bool value = Field<bool>::get(self->oid_, string(field));
                if (value){
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }
            case 'c': GET_FIELD(char, c)
            case 'i': GET_FIELD(int, i)
            case 'h': GET_FIELD(short, h)
            case 'l': GET_FIELD(long, l)        
            case 'I': GET_FIELD(unsigned int, I)
            case 'k': GET_FIELD(unsigned long, k)
            case 'f': GET_FIELD(float, f)
            case 'd': GET_FIELD(double, d)
            case 's': {
                string _s = Field<string>::get(self->oid_, string(field));
                ret = Py_BuildValue("s", _s.c_str());
                break;
            }
            case 'x':
                {                    
                    Id value = Field<Id>::get(self->oid_, string(field));
                    ret = (PyObject*)PyObject_New(_Id, &IdType);
                    ((_Id*)ret)->id_ = value;
                    break;
                }
            case 'y':
                {
                    ObjId value = Field<ObjId>::get(self->oid_, string(field));
                    ret = (PyObject*)PyObject_New(_ObjId, &ObjIdType);
                    ((_ObjId*)ret)->oid_ = value;
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
                vector<string> val = Field< vector<string> >::get(self->oid_, string(field)); 
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
                    vector<Id> value = Field< vector <Id> >::get(self->oid_, string(field));
                    ret = PyTuple_New((Py_ssize_t)value.size());
                    for (unsigned int ii = 0; ii < value.size(); ++ii){
                        _Id * entry = PyObject_New(_Id, &IdType);
                        if (!entry){
                            Py_XDECREF(ret);
                            return NULL;
                        }
                        entry->id_ = value[ii];
                        if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){
                            Py_XDECREF(ret);
                            return NULL;
                        }
                    }
                    break;
                }
            case 'Y': // vector<ObjId>
                {
                    vector<ObjId> value = Field< vector <ObjId> >::get(self->oid_, string(field));
                    ret = PyTuple_New(value.size());
                    for (unsigned int ii = 0; ii < value.size(); ++ii){
                        _ObjId * entry = PyObject_New(_ObjId, &ObjIdType);                       
                        if (!entry){
                            Py_XDECREF(ret);
                            return NULL;
                        }
                        entry->oid_ = value[ii];
                        if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){
                            Py_XDECREF(ret);
                            return NULL;
                        }
                    }
                    break;
                }
                
            default:
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
        
        PyObject * value;
        char * field;
        int ret = 0;
        if (!PyArg_ParseTuple(args, "sO:_pymoose_ObjId_setField", &field,  &value)){
            return NULL;
        }
        char ftype = shortType(getFieldType(self->oid_, string(field), "valueFinfo"));
        
        if (!ftype){
            PyErr_SetString(PyExc_AttributeError, "Field not valid.");
            return NULL;
        }
        
        switch(ftype){
            case 'b':
                {
                    bool _value = (Py_True == value);
                    ret = Field<bool>::set(self->oid_, string(field), _value);
                    break;
                }
            case 'c':
                {
                    char * _value = PyString_AsString(value);
                    if (_value && _value[0]){
                        ret = Field<char>::set(self->oid_, string(field), _value[0]);
                    } else {
                        return NULL;
                    }
                    break;
                }
            case 'i':
                {
                    int _value = PyInt_AsLong(value);
                    if ((_value != -1) || (!PyErr_Occurred())){
                        ret = Field<int>::set(self->oid_, string(field), _value);
                    } else {
                        return NULL;
                    }
                    break;
                }
            case 'h':
                {
                    short _value = (short)PyInt_AsLong(value);
                    if ((_value != -1) || (!PyErr_Occurred())){
                        ret = Field<short>::set(self->oid_, string(field), _value);
                    } else {
                        return NULL;
                    }
                    break;
                }
            case 'l': 
                {
                    long _value = PyInt_AsLong(value);
                    if ((_value != -1) || (!PyErr_Occurred())){
                        ret = Field<long>::set(self->oid_, string(field), _value);
                    } else {
                        return NULL;
                    }
                    break;
                }
            case 'I': 
                {
                    unsigned long _value = PyInt_AsUnsignedLongMask(value);
                    ret = Field<unsigned int>::set(self->oid_, string(field), (unsigned int)_value);
                    break;
                }
            case 'k':
                {
                    unsigned long _value = PyInt_AsUnsignedLongMask(value);
                    ret = Field<unsigned long>::set(self->oid_, string(field), _value);
                    break;
                }
                
            case 'f': 
                {
                    float _value = PyFloat_AsDouble(value);
                    ret = Field<float>::set(self->oid_, string(field), _value);
                    break;
                }
            case 'd': 
                {
                    double _value = PyFloat_AsDouble(value);
                    ret = Field<double>::set(self->oid_, string(field), _value);
                    break;
                }
            case 's': 
                {
                    char * _value = PyString_AsString(value);
                    if (_value){
                        ret = Field<string>::set(self->oid_, string(field), string(_value));
                    } else {
                        return NULL;
                    }
                    break;
                }
            case 'x': // Id
                {
                    if (value){
                        ret = Field<Id>::set(self->oid_, string(field), ((_Id*)value)->id_);
                    } else {
                        PyErr_SetString(PyExc_ValueError, "Null pointer passed as Id value.");
                        return NULL;
                    }
                    break;
                }
            case 'y': // ObjId
                {
                    if (value){
                        ret = Field<ObjId>::set(self->oid_, string(field), ((_ObjId*)value)->oid_);
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
            case 'v': 
                {
                    if (!PySequence_Check(value)){
                        PyErr_SetString(PyExc_TypeError, "For setting vector<int> field, specified value must be a sequence." );
                        return NULL;
                    }
                    Py_ssize_t length = PySequence_Length(value);
                    vector<int> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        int v = PyInt_AsLong(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < int > >::set(self->oid_, string(field), _value);
                    break;
                }
            case 'w': 
                {
                    if (!PySequence_Check(value)){
                        PyErr_SetString(PyExc_TypeError, "For setting vector<short> field, specified value must be a sequence." );
                        return NULL;
                    }
                    Py_ssize_t length = PySequence_Length(value);
                    vector<short> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        short v = PyInt_AsLong(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < short > >::set(self->oid_, string(field), _value);
                    break;
                }
            case 'L': //SET_VECFIELD(long, l)
                {
                    if (!PySequence_Check(value)){
                        PyErr_SetString(PyExc_TypeError, "For setting vector<long> field, specified value must be a sequence." );
                        return NULL;
                    }
                    Py_ssize_t length = PySequence_Length(value);
                    vector<long> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        long v = PyInt_AsLong(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < long > >::set(self->oid_, string(field), _value);
                    break;
                }
            case 'U': //SET_VECFIELD(unsigned int, I)
                {
                    if (!PySequence_Check(value)){
                        PyErr_SetString(PyExc_TypeError, "For setting vector<unsigned int> field, specified value must be a sequence." );
                        return NULL;
                    }
                    Py_ssize_t length = PySequence_Length(value);
                    vector<unsigned int> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        unsigned int v = PyInt_AsUnsignedLongMask(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < unsigned int > >::set(self->oid_, string(field), _value);
                    break;
                }
            case 'K': //SET_VECFIELD(unsigned long, k)
                {
                    if (!PySequence_Check(value)){
                        PyErr_SetString(PyExc_TypeError, "For setting vector<unsigned long> field, specified value must be a sequence." );
                        return NULL;
                    }
                    Py_ssize_t length = PySequence_Length(value);
                    vector<unsigned long> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        unsigned long v = PyInt_AsUnsignedLongMask(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < unsigned long > >::set(self->oid_, string(field), _value);
                    break;
                }
            case 'F': //SET_VECFIELD(float, f)
                {
                    if (!PySequence_Check(value)){
                        PyErr_SetString(PyExc_TypeError, "For setting vector<float> field, specified value must be a sequence." );
                        return NULL;
                    }
                    Py_ssize_t length = PySequence_Length(value);
                    vector<float> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        float v = PyFloat_AsDouble(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < float > >::set(self->oid_, string(field), _value);
                    break;
                }
            case 'D': //SET_VECFIELD(double, d)
                {
                    if (!PySequence_Check(value)){
                        PyErr_SetString(PyExc_TypeError, "For setting vector<double> field, specified value must be a sequence." );
                        return NULL;
                    }
                    Py_ssize_t length = PySequence_Length(value);
                    vector<double> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        double v = PyFloat_AsDouble(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < double > >::set(self->oid_, string(field), _value);
                    break;
                }                
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
                    ret = Field< vector < string > >::set(self->oid_, string(field), _value);
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
    } // _pymoose_Id_setField

    static PyObject * _pymoose_ObjId_getLookupField(_ObjId * self, PyObject * args)
    {
        PyObject * key;
        char * field;
        PyObject * ret;
        if (!PyArg_ParseTuple(args, "sO:_pymoose_ObjId_getLookupField", &field,  &key)){
            return NULL;
        }
        string type = getFieldType(self->oid_, string(field), "lookupFinfo");
        if (type.empty()){
            PyErr_SetString(PyExc_AttributeError, "Field not valid.");
            return NULL;
        }
        vector< string > argType;
        tokenize(type, ",", argType);
        if (argType.size() != 2){
            PyErr_SetString(PyExc_NotImplementedError, "Can handle only single level lookup fields.");
            return NULL;
        }
        char key_type_code = shortType(argType[0]);
        char value_type_code = shortType(argType[1]);
        void * value_ptr = NULL;
        if (value_type_code == 'x'){
            value_ptr = PyObject_New(_Id, &IdType);
        } else if (value_type_code == 'y'){
            value_ptr = PyObject_New(_ObjId, &ObjIdType);
        }
        switch(key_type_code){
            case 'b': {
                ret = lookup_value <bool> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'c': {
                ret = lookup_value <char> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'h': {
                ret = lookup_value <short> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }            
            case 'H': {
                ret = lookup_value <unsigned short> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }            
            case 'i': {
                ret = lookup_value <int> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }            
            case 'I': {
                ret = lookup_value <unsigned int> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }            
            case 'l': {
                ret = lookup_value <long> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }                        
            case 'k': {
                ret = lookup_value <unsigned long> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }                        
            case 'L': {
                ret = lookup_value <long long> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }                        
            case 'K': {
                ret = lookup_value <unsigned long long> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }                        
            case 'd': {
                ret = lookup_value <double> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }                        
            case 'f': {
                ret = lookup_value <float> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'x': {
                ret = lookup_value <Id> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'y': {
                ret = lookup_value <ObjId> (self->oid_, string(field), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            default:
                PyErr_SetString(PyExc_TypeError, "invalid key type");
        }                
        return ret;
    } // _pymoose_ObjId_getLookupField

    static PyObject * _pymoose_ObjId_setLookupField(_ObjId * self, PyObject * args)
    {
        PyObject * key;
        PyObject * value;
        char * field;
        PyObject * ret = NULL;
        if (!PyArg_ParseTuple(args, "sOO:_pymoose_ObjId_getLookupField", &field,  &key, &value)){
            return NULL;
        }
        string type = getFieldType(self->oid_, string(field), "lookupFinfo");
        if (type.empty()){
            PyErr_SetString(PyExc_AttributeError, "Field not valid.");
            return NULL;
        }
        vector< string > argType;
        tokenize(type, ",", argType);
        if (argType.size() != 2){
            PyErr_SetString(PyExc_NotImplementedError, "Can handle only single level lookup fields.");
            return NULL;
        }
        char key_type_code = shortType(argType[0]);
        char value_type_code = shortType(argType[1]);
        switch(key_type_code){
            case 'b': {
                ret = set_lookup_value <bool> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }
            case 'c': {
                ret = set_lookup_value <char> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }
            case 'h': {
                ret = set_lookup_value <short> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }            
            case 'H': {
                ret = set_lookup_value <unsigned short> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }            
            case 'i': {
                ret = set_lookup_value <int> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }            
            case 'I': {
                ret = set_lookup_value <unsigned int> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }            
            case 'l': {
                ret = set_lookup_value <long> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 'k': {
                ret = set_lookup_value <unsigned long> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 'L': {
                ret = set_lookup_value <long long> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 'K': {
                ret = set_lookup_value <unsigned long long> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 'd': {
                ret = set_lookup_value <double> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 'f': {
                ret = set_lookup_value <float> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }
            case 'x': {
                ret = set_lookup_value <Id> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }
            case 'y': {
                ret = set_lookup_value <ObjId> (self->oid_, string(field), value_type_code, key_type_code, key, value);
                break;
            }
            default:
                PyErr_SetString(PyExc_TypeError, "invalid key type");
        }
        return ret;        
    }// _pymoose_ObjId_setLookupField
    
    static PyObject * _pymoose_ObjId_setDestField(_ObjId * self, PyObject * args)
    {
                
        // Minimum number of arguments for setting destFinfo - 1-st
        // the finfo name.
        Py_ssize_t minArgs = 1;
        // Arbitrarily setting maximum on variable argument
        // list. Read: http://www.swig.org/Doc1.3/Varargs.html to
        // understand why
        Py_ssize_t maxArgs = 5;
        PyObject * arglist[5] = {NULL, NULL, NULL, NULL, NULL};
        ostringstream error;
        error << "_pymoose_ObjId_setDestField: ";
        
        if (!PyArg_UnpackTuple(args, "setDestField", minArgs, maxArgs, &arglist[0], &arglist[1], &arglist[2], &arglist[3], arglist[4])){
            return NULL;
        }
        char * fieldName = PyString_AsString(arglist[0]);
        if (!fieldName){ // not a string, raises TypeError
            return NULL;
        }
        string type = getFieldType(self->oid_, string(fieldName), "destFinfo");
        if (type.empty()){
            error << "No such function field available";
            PyErr_SetString(PyExc_ValueError, error.str().c_str());
            return NULL;
        }
        vector< string > argType;
        tokenize(type, ",", argType);
        if (argType.empty()){
            error << "Error in determining the data type of the arguments: received: " << type
                  << "Expected " << type;
            PyErr_SetString(PyExc_RuntimeError, error.str().c_str());
            return NULL;
        } else if (argType.size() >= maxArgs) {
            error << "_pymoose_ObjId_setDestField: number of arguments to this function exceeds the implemented maximum=" << (maxArgs - 1)
                  << ".\nExpected arguments: " << type;
            PyErr_SetString(PyExc_NotImplementedError, error.str().c_str());
            return NULL;
        }
        ostringstream argstream;
        for (size_t ii = 0; ii < argType.size(); ++ii){
            PyObject * arg = arglist[ii+1];
            if ( arg == NULL){
                bool ret = SetGet0::set(self->oid_, string(fieldName));
                if (ret){
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }
            switch (shortType(argType[ii])){                    
                case 'c':
                    {
                        char * param = PyString_AsString(arg);
                        if (!param){
                            error << ii << "-th expected of type char/string";
                            PyErr_SetString(PyExc_TypeError, error.str().c_str());
                            return NULL;
                        } else if (strlen(param) == 0){
                            error << "Empty string not allowed.";
                            PyErr_SetString(PyExc_ValueError, error.str().c_str());
                            return NULL;
                        }
                        argstream << param[0] << ",";
                    }
                    break;
                case 'i': case 'l':
                    {
                        long param = PyInt_AsLong(arg);
                        if (param == -1 && PyErr_Occurred()){
                            return NULL;
                        }
                        argstream << param << ",";
                    }
                    break;
                case 'I': case 'k':
                    {
                        unsigned long param =PyLong_AsUnsignedLong(arg);
                        if (PyErr_Occurred()){
                            return NULL;
                        }
                        argstream << param << ",";                            
                    }
                    break;
                case 'f': case 'd':
                    {
                        double param = PyFloat_AsDouble(arg);
                        argstream << param << ",";
                    }
                    break;
                case 's':
                    {
                        char * param = PyString_AsString(arg);
                        argstream << string(param) << ",";
                    }
                    break;
                    // We do handle multiple vectors. Use the argument
                    // list as a single vector argument.
                case 'v': 
                    {
                        return _set_vector_destFinfo<int>(self, string(fieldName), ii, arg);
                        break;
                    }
                case 'w': 
                    {
                        return _set_vector_destFinfo<short>(self, string(fieldName), ii, arg);
                        break;
                    }
                case 'L': //SET_VECFIELD(long, l)
                    {
                        return _set_vector_destFinfo<long>(self, string(fieldName), ii, arg);
                        break;
                    }
                case 'U'://SET_VECFIELD(unsigned int, I)
                    {
                        return _set_vector_destFinfo<unsigned int>(self, string(fieldName), ii, arg);
                        break;
                    }
                case 'K': //SET_VECFIELD(unsigned long, k)
                    {
                        return _set_vector_destFinfo<unsigned long>(self, string(fieldName), ii, arg);
                        break;
                    }
                case 'F': //SET_VECFIELD(float, f)
                    {
                        return _set_vector_destFinfo<float>(self, string(fieldName), ii, arg);
                        break;
                    }
                case 'D': //SET_VECFIELD(double, d)
                    {
                        return _set_vector_destFinfo<double>(self, string(fieldName), ii, arg);
                        break;
                    }                
                case 'S':
                    {
                        return _set_vector_destFinfo<string>(self, string(fieldName), ii, arg);
                        break;
                    }
                case 'X':
                    {
                        return _set_vector_destFinfo<Id>(self, string(fieldName), ii, arg);
                        break;
                    }
                case 'Y':
                    {
                        return _set_vector_destFinfo<ObjId>(self, string(fieldName), ii, arg);
                    }
                default:
                    {
                        error << "Cannot handle argument type: " << argType[ii];
                        PyErr_SetString(PyExc_TypeError, error.str().c_str());
                        return NULL;
                    }
            } // switch (shortType(argType[ii])
        } // for (size_t ii = 0; ...
        // TODO: handle vector args and void functions properly
        string argstring = argstream.str();
        if (argstring.length() < 2 ){
            error << "Could not find any valid argument. Giving up.";
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            return NULL;
        }
        
        argstring = argstring.substr(0, argstring.length() - 1);        
        bool ret = SetGet::strSet(self->oid_, string(fieldName), argstring);
        if (ret){
            Py_RETURN_TRUE;
        } else {
            Py_RETURN_FALSE;
        }        
    } // _pymoose_ObjId_setDestField
    // 2011-03-23 15:28:26 (+0530)
    static PyObject * _pymoose_ObjId_getFieldNames(_ObjId * self, PyObject *args)
    {
        char * ftype = NULL;
        if (!PyArg_ParseTuple(args, "|s:_pymoose_ObjId_getFieldNames", &ftype)){
            return NULL;
        }
        string ftype_str = (ftype != NULL)? string(ftype): "";
        vector<string> ret;
        if (ftype_str == ""){
            for (const char **a = &FINFOTYPES[0]; *a; ++a){
                vector<string> fields = getFieldNames(self->oid_, string(*a));
                ret.insert(ret.end(), fields.begin(), fields.end());
            }            
        } else {
            ret = getFieldNames(self->oid_, ftype_str);
        }
        
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

    static PyObject * _pymoose_ObjId_getNeighbors(_ObjId * self, PyObject * args)
    {
        char * field = NULL;
        if (!PyArg_ParseTuple(args, "s:_pymoose_ObjId_getNeighbors", &field)){
            return NULL;
        }
        vector< Id > val = LookupField< string, vector< Id > >::get(self->oid_, "neighbours", string(field));
    
        PyObject * ret = PyTuple_New((Py_ssize_t)val.size());
        for (unsigned int ii = 0; ii < val.size(); ++ ii ){            
            _Id * entry = PyObject_New(_Id, &IdType);
            if (!entry || PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){ 
                Py_XDECREF(ret);                                  
                ret = NULL;                                 
                break;                                      
            }
            entry->id_ = val[ii];
        }
        return ret;
    }

    // 2011-03-28 10:10:19 (+0530)
    // 2011-03-23 15:13:29 (+0530)
    // getChildren is not required as it can be accessed as getField("children")

    // 2011-03-28 10:51:52 (+0530)
    static PyObject * _pymoose_ObjId_connect(_ObjId * self, PyObject * args)
    {
        PyObject * destPtr = NULL;
        char * srcField, * destField, * msgType;

        if(!PyArg_ParseTuple(args, "sOss:_pymoose_ObjId_connect", &srcField, &destPtr, &destField, &msgType)){
            return NULL;
        }
        _ObjId * dest = reinterpret_cast<_ObjId*>(destPtr);
        bool ret = (getShell().doAddMsg(msgType, self->oid_, string(srcField), dest->oid_, string(destField)) != Msg::bad);
        if (!ret){
            PyErr_SetString(PyExc_NameError, "connect failed: check field names and type compatibility.");
            return NULL;
        }
        return Py_BuildValue("i", ret);
    }

    static PyObject * _pymoose_ObjId_richCompare(_ObjId * self, PyObject * other, int op)
    {
        int ret;
        if (!self || !other){
            ret = 0;
        } else if (!ObjId_SubtypeCheck(other)){
            ret = 0;
        } else if (op == Py_EQ){
            ret = (self->oid_ == ((_ObjId*)other)->oid_);
        } else if (op == Py_NE){
            ret = !(self->oid_ == ((_ObjId*)other)->oid_);
        } else {
            ret = 0;
        }
        return Py_BuildValue("i", ret);
    }

    static PyObject * _pymoose_ObjId_getDataIndex(_ObjId * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":_pymoose_ObjId_getDataIndex")){
            return NULL;
        }
        PyObject * ret = Py_BuildValue("I", self->oid_.dataId.value());
        return ret;
    }

    // WARNING: fieldIndex has been deprecated in dh_branch. This
    // needs to be updated accordingly.  The current code is just
    // place-holer to avoid compilation errors.
    static PyObject * _pymoose_ObjId_getFieldIndex(_ObjId * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":_pymoose_ObjId_getFieldIndex")){
            return NULL;
        }
        PyObject * ret = Py_BuildValue("I", self->oid_.dataId.value());
        return ret;
    }
    
    ////////////////////////////////////////////
    // The following are global functions
    ////////////////////////////////////////////

    
    static PyObject * _pymoose_copy(PyObject * dummy, PyObject * args, PyObject * kwargs)
    {
        PyObject * src, *dest;
        char * newName;
        static const char * kwlist[] = {"src", "dest", "name", "n", "toGlobal", "copyExtMsg", NULL};
        unsigned int num=1, toGlobal=0, copyExtMsgs=0;
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOs|III:_pymoose_copy", const_cast<char**>(kwlist), &src, &dest, &newName, &num, &toGlobal, &copyExtMsgs)){
            return NULL;
        }
        if (!Id_SubtypeCheck(src)){
            PyErr_SetString(PyExc_TypeError, "Source must be instance of Id.");
            return NULL;
        } else if (!Id_SubtypeCheck(dest)){
            PyErr_SetString(PyExc_TypeError, "Destination must be instance of Id.");
            return NULL;
        }
        if (((_Id*)src)->id_ == Id()){
            PyErr_SetString(PyExc_ValueError, "Cannot make copy of moose shell.");
            return NULL;
        }
        _Id * tgt = PyObject_New(_Id, &IdType);
        tgt->id_ = getShell().doCopy(((_Id*)src)->id_, ((_Id*)dest)->id_, string(newName), num, toGlobal, copyExtMsgs);
        PyObject * ret = (PyObject*)tgt;
        return ret;            
    }

    // Not sure what this function should return... ideally the Id of the
    // moved object - does it change though?
    static PyObject * _pymoose_move(PyObject * dummy, PyObject * args)
    {
        PyObject * src, * dest;
        if (!PyArg_ParseTuple(args, "OO:_pymoose_move", &src, &dest)){
            return NULL;
        }
        if (((_Id*)src)->id_ == Id()){
            PyErr_SetString(PyExc_ValueError, "Cannot move moose shell");
            return NULL;
        }
        getShell().doMove(((_Id*)src)->id_, ((_Id*)dest)->id_);
        Py_RETURN_NONE;
    }

    static PyObject * _pymoose_delete(PyObject * dummy, PyObject * args)
    {
        PyObject * obj;
        if (!PyArg_ParseTuple(args, "O:_pymoose_delete", &obj)){
            return NULL;
        }
        if (((_Id*)obj)->id_ == Id()){
            PyErr_SetString(PyExc_ValueError, "Cannot delete moose shell.");
            return NULL;
        }
        getShell().doDelete(((_Id*)obj)->id_);
        ((_Id*)obj)->id_ = Id();
        Py_RETURN_NONE;
    }

    static PyObject * _pymoose_useClock(PyObject * dummy, PyObject * args)
    {
        char * path, * field;
        unsigned int tick;
        if(!PyArg_ParseTuple(args, "Iss:_pymoose_useClock", &tick, &path, &field)){
            return NULL;
        }
        getShell().doUseClock(string(path), string(field), tick);
        Py_RETURN_NONE;
    }
    static PyObject * _pymoose_setClock(PyObject * dummy, PyObject * args)
    {
        unsigned int tick;
        double dt;
        if(!PyArg_ParseTuple(args, "Id:_pymoose_setClock", &tick, &dt)){
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
        if(!PyArg_ParseTuple(args, "d:_pymoose_start", &runtime)){
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

    static PyObject * _pymoose_exists(PyObject * dummy, PyObject * args)
    {
        char * path;
        if (!PyArg_ParseTuple(args, "s", &path)){
            return NULL;
        }
        return Py_BuildValue("i", Id(path) != Id() || string(path) == "/" || string(path) == "/root");
    }
    
    static PyObject * _pymoose_loadModel(PyObject * dummy, PyObject * args)
    {
        char * fname = NULL, * modelpath = NULL, * solverclass = NULL;
        if(!PyArg_ParseTuple(args, "ss|s:_pymoose_loadModel", &fname, &modelpath, &solverclass)){
            return NULL;
        }
        _Id * model = (_Id*)PyObject_New(_Id, &IdType);
        if (!solverclass){
            model->id_ = getShell().doLoadModel(string(fname), string(modelpath));
        } else {
            model->id_ = getShell().doLoadModel(string(fname), string(modelpath), string(solverclass));
        }
        PyObject * ret = reinterpret_cast<PyObject*>(model);
        return ret;
    }

    static PyObject * _pymoose_setCwe(PyObject * dummy, PyObject * args)
    {
        PyObject * element = NULL;
        const char * path = "/";
        Id id;
        if(PyArg_ParseTuple(args, "s:_pymoose_setCwe", const_cast<char**>(&path))){
            id = Id(string(path));
        } else if (PyArg_ParseTuple(args, "O:_pymoose_setCwe", &element)){
            PyErr_Clear();
            if (Id_SubtypeCheck(element)){
                id = (reinterpret_cast<_Id*>(element))->id_;
            } else if (ObjId_SubtypeCheck(element)){
                id = (reinterpret_cast<_ObjId*>(element))->oid_.id;                    
            } else {
                PyErr_SetString(PyExc_NameError, "setCwe: Argument must be an Id or ObjId");
                return NULL;
            }
        } else {
            return NULL;
        }
        getShell().setCwe(id);
        Py_RETURN_NONE;
    }

    static PyObject * _pymoose_getCwe(PyObject * dummy, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":_pymoose_getCwe")){
            return NULL;
        }
        _Id * cwe = (_Id*)PyObject_New(_Id, &IdType);
        cwe->id_ = getShell().getCwe();        
        PyObject * ret = (PyObject*)cwe;
        return ret;
    }

    static PyObject * _pymoose_connect(PyObject * dummy, PyObject * args)
    {
        PyObject * srcPtr = NULL, * destPtr = NULL;
        char * srcField = NULL, * destField = NULL, * msgType = NULL;
        if(!PyArg_ParseTuple(args, "OsOss:_pymoose_connect", &srcPtr, &srcField, &destPtr, &destField, &msgType)){
            return NULL;
        }
        _ObjId * dest = reinterpret_cast<_ObjId*>(destPtr);
        _ObjId * src = reinterpret_cast<_ObjId*>(srcPtr);
        bool ret = (getShell().doAddMsg(msgType, src->oid_, string(srcField), dest->oid_, string(destField)) != Msg::bad);
        if (!ret){
            PyErr_SetString(PyExc_NameError, "connect failed: check field names and type compatibility.");
            return NULL;
        }
        return Py_BuildValue("i", ret);        
    }

    static PyObject * _pymoose_getFieldDict(PyObject * dummy, PyObject * args)
    {
        char * className = NULL;
        char * fieldType = NULL;
        if (!PyArg_ParseTuple(args, "s|s:_pymoose_getFieldDict", &className, &fieldType)){
            return NULL;
        }
        if (!className || (strlen(className) <= 0)){
            PyErr_SetString(PyExc_ValueError, "Expected non-empty class name.");
            return NULL;
        }
        
        Id classId = Id("/classes/" + string(className));
        if (classId == Id()){
            string msg = string(className);
            msg += " not a valid MOOSE class.";
            PyErr_SetString(PyExc_NameError, msg.c_str());
            return NULL;
        }
        static const char * finfoTypes [] = {"valueFinfo", "lookupFinfo", "srcFinfo", "destFinfo", "sharedFinfo", NULL};
        vector <string> fields, types;
        if (fieldType && strlen(fieldType) > 0){
            if (inner_getFieldDict(classId, string(fieldType), fields, types) == 0){
                PyErr_SetString(PyExc_ValueError, "Invalid finfo type.");
                return NULL;
            }
        } else {
            for (const char ** ptr = finfoTypes; *ptr != NULL; ++ptr){
                if (inner_getFieldDict(classId, string(*ptr), fields, types) == 0){
                    string message = "No such finfo type: ";
                    message += string(*ptr);
                    PyErr_SetString(PyExc_ValueError, message.c_str());
                    return NULL;
                }
            }
        }
        PyObject * ret = PyDict_New();
        if (!ret){
            PyErr_SetString(PyExc_SystemError, "Could not allocate dictionary object.");
            return NULL;
        }
        for (unsigned int ii = 0; ii < fields.size(); ++ ii){
            if (PyDict_SetItemString(ret, fields[ii].c_str(), Py_BuildValue("s", types[ii].c_str())) == -1){
                Py_XDECREF(ret);
                return NULL;
            }
        }
        return ret;
    }

    int inner_getFieldDict(Id classId, string finfoType, vector<string>& fieldNames, vector<string>&fieldTypes)
    {
        unsigned int numFinfos = Field<unsigned int>::get(ObjId(classId), "num_" + string(finfoType));
        Id fieldId(classId.path() + "/" + string(finfoType));
        if (fieldId == Id()){
            return 0;
        }
        for (unsigned int ii = 0; ii < numFinfos; ++ii){
            string fieldName = Field<string>::get(ObjId(fieldId, DataId(0, ii, 0)), "name");
            fieldNames.push_back(fieldName);
            string fieldType = Field<string>::get(ObjId(fieldId, DataId(0, ii, 0)), "type");
            fieldTypes.push_back(fieldType);
        }
        return 1;
    }

    PyObject * _pymoose_syncDataHandler(PyObject * dummy, _Id * target)
    {
        getShell().doSyncDataHandler(target->id_);
        Py_RETURN_NONE;
    }


    PyObject * _pymoose_seed(PyObject * dummy, PyObject * args)
    {
        long seed = 0;
        if (!PyArg_ParseTuple(args, "|l", &seed)){
            return NULL;
        }
        mtseed(seed);
        Py_RETURN_NONE;
    }

    PyObject * _pymoose_wildcardFind(PyObject * dummy, PyObject * args)
    {
        vector <Id> objects;
        char * wildcard_path = NULL;
        if (!PyArg_ParseTuple(args, "s", &wildcard_path)){
            return NULL;
        }
        getShell().wildcard(string(wildcard_path), objects);
        PyObject * ret = PyTuple_New(objects.size());
        for (unsigned int ii = 0; ii < objects.size(); ++ii){
            _Id * entry = PyObject_New(_Id, &IdType);                       
            if (!entry){
                Py_XDECREF(ret);
                return NULL;
            }
            entry->id_ = objects[ii];
            if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){
                Py_XDECREF(ret);
                return NULL;
            }
        }
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
