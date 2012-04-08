// moosemodule.h --- 
// 
// Filename: moosemodule.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Thu Mar 10 17:11:06 2011 (+0530)
// Version: 
// Last-Updated: Sun Apr  8 17:07:09 2012 (+0530)
//           By: subha
//     Update #: 603
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
    static PyObject * _pymoose_ObjId_getattro(_ObjId * self, PyObject * attr);
    static PyObject * _pymoose_ObjId_getField(_ObjId * self, PyObject * args);
    static int _pymoose_ObjId_setattro(_ObjId * self, PyObject * attr, PyObject * value);
    static PyObject * _pymoose_ObjId_setField(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getLookupField(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_setLookupField(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_setDestField(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getFieldNames(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getFieldType(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getDataIndex(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getFieldIndex(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getNeighbors(_ObjId * self, PyObject * args);
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
    static PyObject * _pymoose_getField(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_syncDataHandler(PyObject * dummy, _Id * target);
    static PyObject * _pymoose_seed(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_wildcardFind(PyObject * dummy, PyObject * args);
    // This should not be required or accessible to the user. Put here
    // for debugging threading issue.
    static PyObject * _pymoose_quit(PyObject * dummy);
    PyMODINIT_FUNC init_moose();


    int inner_getFieldDict(Id classId, string finfoType, vector<string>& fields, vector<string>& types); 


    
} //!extern "C"

/// push_item functions convert a Python object to an appropriate C++
/// object type and push it into a supplied vector.
template <class A> inline bool pyobj_to_cpp(A& target, PyObject * item)
{
    return false;
}
template <> inline bool pyobj_to_cpp<int>(int& ref, PyObject * item)
{
    int v = PyInt_AsLong(item);
    ref = v;
    return true;
}
template <> inline bool pyobj_to_cpp<long>(long& ref, PyObject * item)
{
    long v = PyInt_AsLong(item);
    ref = v;
    return true;
}
template <> inline bool pyobj_to_cpp<short>(short& ref, PyObject * item)
{
    short v = PyInt_AsLong(item);
    ref = v;
    return true;
}
template <> inline bool pyobj_to_cpp<float>(float& ref, PyObject * item)
{
    float v = (float)PyFloat_AsDouble(item);
    if ( v == -1.0 && PyErr_Occurred()){
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of floating point numbers.");
        return false;
    }
    ref = v;
    return true;
}
template <> inline bool pyobj_to_cpp<double>(double& ref, PyObject * item)
{
    double v = PyFloat_AsDouble(item);
    if ( v == -1.0 && PyErr_Occurred()){
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of floating point numbers.");
        return false;
    }
    ref = v;
    return true;
}
template <> inline bool pyobj_to_cpp<string>(string& ref, PyObject * item)
{
    char* tmp = PyString_AsString(item);
    if (tmp == NULL){
        return false;
    }
    ref = string(tmp);
    return true;
}
template <> inline bool pyobj_to_cpp<unsigned int>(unsigned int& ref, PyObject * item)
{
    unsigned int v = PyInt_AsUnsignedLongMask(item);
    ref = v;
    return true;
}
template <> inline bool pyobj_to_cpp<unsigned long>(unsigned long& ref, PyObject * item)
{
    unsigned long v = PyInt_AsUnsignedLongMask(item);
    ref = v;
    return true;
}

template <> inline bool pyobj_to_cpp<Id>(Id& ref, PyObject * item)
{
    _Id * value = (_Id*)item;
    if (value != NULL){
        ref = value->id_;
        return true;
    }
    return false;
}

template <> inline bool pyobj_to_cpp<ObjId>(ObjId& ref, PyObject * item)
{
    _ObjId * value = (_ObjId*)item;
    if (value != NULL){
        ref = value->oid_;
        return true;
    }
    return false;
}

/// Convert a Python sequence to a C++ vector
template <class A> bool pysequence_to_vector(vector <A> & store, PyObject * sequence)
{
    Py_ssize_t length = PySequence_Length(sequence);
    A value;
    for (unsigned int ii = 0; ii < length; ++ii){
        PyObject * item = PySequence_GetItem(sequence, ii);
        if (item == NULL){
            ostringstream error;
            error << "Item # " << ii << " is NULL";
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            return false;
        }
        if (pyobj_to_cpp<A>(value, item) == false){
            ostringstream error;
            error << "Cannot handle sequence of type " << typeid(A).name();
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            return false;
        } else {
            store.push_back(value);
        }            
    }
    return true;
}
/// Set a destinfo that takes a vector argument
template <class A> inline PyObject* _set_vector_destFinfo(_ObjId* obj, string fieldName, int argIndex, PyObject * value)
{
    ostringstream error;
    if (!PySequence_Check(value)){                                  
        PyErr_SetString(PyExc_TypeError, "For setting vector field, specified value must be a sequence." );
        return NULL;
    }
    if (argIndex > 0){
        PyErr_SetString(PyExc_TypeError, "Can handle only single-argument functions with vector argument." );
        return NULL;
    }            
    vector<A> _value;
    if (!pysequence_to_vector(_value, value)){
        return NULL;
    }
    bool ret = SetGet1< vector < A > >::set(obj->oid_, fieldName, _value);
    if (ret){
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

/// The circus with value_ptr is to allow Id, ObjId to be allocated.
template <class KeyType> inline PyObject * lookup_value(const ObjId& oid, string fname, char value_type_code, char key_type_code, PyObject * key, void * value_ptr=NULL)
{
    PyObject * ret = NULL;
    KeyType cpp_key;
    if (!pyobj_to_cpp<KeyType>(cpp_key, key)){
        return NULL;
    }
    string value_type_str = string(1, value_type_code);
    switch (value_type_code){
        case 'b': { // boolean is a special case that PyBuildValue does not handle
            bool value = LookupField < KeyType, bool > ::get(oid, fname, cpp_key);
            if (value){
                Py_RETURN_TRUE;
            } else {
                Py_RETURN_FALSE;
            }
        }
        case 'c': {
            char value = LookupField < KeyType, char > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }            
        case 'h': {
            short value = LookupField < KeyType, short > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }            
        case 'H': {
            unsigned short value = LookupField < KeyType, unsigned short > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }            
        case 'i': {
            int value = LookupField < KeyType, int > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }            
        case 'I': {
            unsigned int value = LookupField < KeyType, unsigned int > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }            
        case 'l': {
            long value = LookupField < KeyType, long > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }                        
        case 'k': {
            unsigned long value = LookupField < KeyType, unsigned long > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }                        
        case 'L': {
            long long value = LookupField < KeyType, long long > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }                        
        case 'K': {
            unsigned long long value = LookupField < KeyType, unsigned long long > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }                        
        case 'd': {
            double value = LookupField < KeyType, double > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }                        
        case 'f': {
            float value = LookupField < KeyType, float > ::get(oid, fname, cpp_key);
            ret = Py_BuildValue(value_type_str.c_str(), value);
            break;
        }
        case 'x': {
            assert(value_ptr != NULL);
            ((_Id*)value_ptr)->id_ = LookupField < KeyType, Id > ::get(oid, fname, cpp_key);
            ret = (PyObject*)value_ptr;
            break;
        }
        case 'y': {
            assert(value_ptr != NULL);
            ((_ObjId*)value_ptr)->oid_ = LookupField < KeyType, Id > ::get(oid, fname, cpp_key);
            ret = (PyObject*)value_ptr;
            break;
        }
        default:
            PyErr_SetString(PyExc_TypeError, "invalid value type");
            return NULL;
    }
    return ret;
}
/// The circus with value_ptr is to allow Id, ObjId to be allocated.
template <class KeyType> inline PyObject * set_lookup_value(const ObjId& oid, string fname, char value_type_code, char key_type_code, PyObject * key, PyObject * value_obj)
{
    bool success = false;
    KeyType cpp_key;
    if (!pyobj_to_cpp<KeyType>(cpp_key, key)){
        return NULL;
    }
    string value_type_str = string(1, value_type_code);
    switch (value_type_code){
        case 'b': { // boolean is a special case that PyBuildValue does not handle
            bool value;
            if (!pyobj_to_cpp<bool>(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, bool > ::set(oid, fname, cpp_key, value);
            break;
        }
        case 'c': {
            char value;
            if (!pyobj_to_cpp<char>(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, char > ::set(oid, fname, cpp_key, value);
            break;
        }            
        case 'h': {
            short value;
            if (!pyobj_to_cpp<short>(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, short > ::set(oid, fname, cpp_key, value);
            break;
        }            
        case 'H': {
            unsigned short value;
            if (!pyobj_to_cpp< unsigned short >(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, unsigned short > ::set(oid, fname, cpp_key, value);
            break;
        }            
        case 'i': {
            int value;
            if (!pyobj_to_cpp< int >(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, int > ::set(oid, fname, cpp_key, value);
            break;
        }            
        case 'I': {
            unsigned int value;
            if (!pyobj_to_cpp< unsigned int >(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, unsigned int > ::set(oid, fname, cpp_key, value);
            break;
        }            
        case 'l': {
            long value;
            if (!pyobj_to_cpp< long >(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, long > ::set(oid, fname, cpp_key, value);
            break;
        }                        
        case 'k': {
            unsigned long value;
            if (!pyobj_to_cpp< unsigned long >(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, unsigned long > ::set(oid, fname, cpp_key, value);
            break;
        }                        
        case 'L': {
            long long value;
            if (!pyobj_to_cpp< long long >(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, long long > ::set(oid, fname, cpp_key, value);
            break;
        }                        
        case 'K': {
            unsigned long long value;
            if (!pyobj_to_cpp< unsigned long long >(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, unsigned long long > ::set(oid, fname, cpp_key, value);
            break;
        }                        
        case 'd': {
            double value;
            if (!pyobj_to_cpp< double >(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, double > ::set(oid, fname, cpp_key, value);
            break;
        }                        
        case 'f': {
            float value;
            if (!pyobj_to_cpp< float >(value, value_obj)){
                return NULL;
            }
            success = LookupField < KeyType, float > ::set(oid, fname, cpp_key, value);
            break;
        }
        case 'x': {
            assert(value_obj!=NULL);
            Id value = ((_Id*)value_obj)->id_;
            success = LookupField < KeyType, Id > ::set(oid, fname, cpp_key, value);
            break;
        }
        case 'y': {
            assert(value_obj!=NULL);
            ObjId value = ((_ObjId*)value_obj)->oid_;
            success = LookupField < KeyType, ObjId > ::set(oid, fname, cpp_key, value);
            break;
        }
        default:
            PyErr_SetString(PyExc_TypeError, "invalid value type");
            return NULL;
    }
    if (success){
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

#endif // _MOOSEMODULE_H

// 
// moosemodule.h ends here
