// moosemodule.h --- 
// 
// Filename: moosemodule.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Thu Mar 10 17:11:06 2011 (+0530)
// Version: 
// Last-Updated: Thu Feb  9 15:05:48 2012 (+0530)
//           By: Subhasis Ray
//     Update #: 345
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change log:
// 
// 
// 

// Code:
#ifndef _MOOSEMODULE_H
#define _MOOSEMODULE_H

#include <string>
#include "../basecode/Id.h"
extern "C" {
    /**
       _Id wraps the Id class - where each element is identified by Id
    */
    typedef struct {
        PyObject_HEAD
        Id id_;
    } _Id;
    /**
       _ObjId wraps the subelements of a Id - identified by
       index. This is different from the Element in GENESIS
       terminology. Since Neutral is now by default an array element,
       we call the individual entries in it as Elements.

       According to MOOSE API, ObjId is a composition of Id and DataId
       - thus uniquely identifying a field of a subelement of Neutral.

       Since the individual subelements are identified by their ObjId
       only (there is no intermediate id for the subelements except
       ObjId with DataId(index, 0) ), we use ObjId for recognizing
       both.
    */
    typedef struct {
        PyObject_HEAD
        ObjId oid_;
    } _ObjId;

    static PyObject * MooseError;
    //////////////////////////////////////////
    // Methods for Id class
    //////////////////////////////////////////
    static int _pymoose_Id_init(_Id * self, PyObject * args, PyObject * kwargs);
    static long _pymoose_Id_hash(_Id * self, PyObject * args);
    
    static void _pymoose_Id_dealloc(_Id * self);
    static PyObject * _pymoose_Id_repr(_Id * self);
    static PyObject * _pymoose_Id_str(_Id * self);
    static PyObject * _pymoose_Id_delete(_Id * self, PyObject * args);
    static PyObject * _pymoose_Id_getValue(_Id * self, PyObject * args);
    static PyObject * _pymoose_Id_getPath(_Id * self, PyObject * args);
    /* Id functions to allow part of sequence protocol */
    static Py_ssize_t _pymoose_Id_getLength(_Id * self);
    static PyObject * _pymoose_Id_getItem(_Id * self, Py_ssize_t index);
    static PyObject * _pymoose_Id_getSlice(_Id * self, PyObject * args);    
    static PyObject * _pymoose_Id_getShape(_Id * self, PyObject * args);    
    static PyObject * _pymoose_Id_richCompare(_Id * self, PyObject * args, int op);
    static int _pymoose_Id_contains(_Id * self, PyObject * args);
    ///////////////////////////////////////////
    // Methods for ObjId class
    ///////////////////////////////////////////
    static int _pymoose_ObjId_init(_ObjId * self, PyObject * args, PyObject * kwargs);
    static long _pymoose_ObjId_hash(_ObjId * self, PyObject * args);
    static void _pymoose_ObjId_dealloc(_ObjId * self);
    static PyObject * _pymoose_ObjId_repr(_ObjId * self);
    // static PyObject * _pymoose_ObjId_str(_ObjId * self);
    static PyObject * _pymoose_ObjId_getField(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_setField(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_setDestField(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getFieldNames(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getFieldType(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getDataIndex(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getFieldIndex(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getNeighbours(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getId(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_connect(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_richCompare(_ObjId * self, PyObject * args, int op);
    
    ////////////////////////////////////////////////
    // static functions to be accessed from Python
    ////////////////////////////////////////////////


    // The following are global functions
    static PyObject * _pymoose_useClock(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_setClock(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_start(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_reinit(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_stop(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_isRunning(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_exists(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_loadModel(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_setCwe(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_getCwe(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_copy(PyObject * dummy, PyObject * args, PyObject * kwargs);
    static PyObject * _pymoose_move(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_delete(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_connect(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_getFieldDict(PyObject * dummy, PyObject * args);    
    static PyObject * _pymoose_syncDataHandler(PyObject * dummy, _Id * target);
    static PyObject * _pymoose_seed(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_wildcardFind(PyObject * dummy, PyObject * args);
    PyMODINIT_FUNC init_moose();


    int inner_getFieldDict(Id classId, string finfoType, vector<string>& fields, vector<string>& types); 


    
} //!extern "C"

template <class A> inline PyObject * get_item(vector<A>& store, PyObject * item, int index)
{
    Py_RETURN_FALSE;
}
template <> inline PyObject * get_item<int>(vector<int>& store, PyObject * item, int index)
{
    int v = PyInt_AsLong(item);
    store.push_back(v);
    Py_RETURN_TRUE;
}
template <> inline PyObject * get_item<long>(vector<long>& store, PyObject * item, int index)
{
    long v = PyInt_AsLong(item);
    store.push_back(v);
    Py_RETURN_TRUE;
}
template <> inline PyObject * get_item<short>(vector<short>& store, PyObject * item, int index)
{
    short v = PyInt_AsLong(item);
    store.push_back(v);
    Py_RETURN_TRUE;
}
template <> inline PyObject * get_item<float>(vector<float>& store, PyObject * item, int index)
{
    float v = (float)PyFloat_AsDouble(item);
    if ( v == -1.0 && PyErr_Occurred()){
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of floating point numbers.");
        Py_RETURN_FALSE;
    }
    store.push_back(v);
    Py_RETURN_TRUE;
}
template <> inline PyObject * get_item<double>(vector<double>& store, PyObject * item, int index)
{
    double v = PyFloat_AsDouble(item);
    if ( v == -1.0 && PyErr_Occurred()){
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of floating point numbers.");
        Py_RETURN_FALSE;
    }
    store.push_back(v);
    Py_RETURN_TRUE;
}
template <> inline PyObject * get_item<string>(vector<string>& store, PyObject * item, int index)
{
    char* tmp = PyString_AsString(item);
    if (tmp == NULL){
        Py_RETURN_FALSE;
    }
    store.push_back(string(tmp));
    Py_RETURN_TRUE;
}
template <> inline PyObject * get_item<unsigned int>(vector<unsigned int>& store, PyObject * item, int index)
{
    unsigned int v = PyInt_AsUnsignedLongMask(item);
    store.push_back(v);
    Py_RETURN_TRUE;
}
template <> inline PyObject * get_item<unsigned long>(vector<unsigned long>& store, PyObject * item, int index)
{
    unsigned long v = PyInt_AsUnsignedLongMask(item);
    store.push_back(v);
    Py_RETURN_TRUE;
}

template <class A> inline PyObject* _set_vector_destFinfo(_ObjId* obj, string fieldName, int argIndex, PyObject * value)
{
    ostringstream error;
    if (!PySequence_Check(value)){                                  
        PyErr_SetString(PyExc_TypeError, "For setting vector field, specified value must be a sequence." );
        return false;
    }
    if (argIndex > 0){
        PyErr_SetString(PyExc_TypeError, "Can handle only single-argument functions with vector argument." );
        return false;
    }            
    Py_ssize_t length = PySequence_Length(value);
    vector<A> _value;
    for (unsigned int ii = 0; ii < length; ++ii){
        PyObject * item = PySequence_GetItem(value, ii);
        if (item == NULL){
            error << "Item # " << ii << " is NULL";
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            Py_RETURN_FALSE;
        }
        if (get_item<A>(_value, item, ii) == Py_False){
            error << "Cannot handle sequence of type " << typeid(A).name();
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            Py_RETURN_FALSE;
        }
    }
    bool ret = SetGet1< vector < A > >::set(obj->oid_, fieldName, _value);
    if (ret){
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

#endif // _MOOSEMODULE_H

// 
// moosemodule.h ends here
