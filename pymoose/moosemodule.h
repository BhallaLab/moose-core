// moosemodule.h --- 
// 
// Filename: moosemodule.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Thu Mar 10 17:11:06 2011 (+0530)
// Version: 
// Last-Updated: Sat Jun 29 15:21:13 2013 (+0530)
//           By: subha
//     Update #: 1129
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
#if PY_MAJOR_VERSION >= 3
    // int has been replaced by long
#define PyInt_Check PyLong_Check
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AsUnsignedLongMask PyLong_AsUnsignedLongMask
#define PyInt_FromLong PyLong_FromLong
  // string has been replaced by unicode
#define PyString_Check PyUnicode_Check
#define PyString_FromString PyUnicode_FromString
#define PyString_FromFormat PyUnicode_FromFormat
#define PyString_AsString(str)\
  PyBytes_AS_STRING(PyUnicode_AsEncodedString(str, "utf-8", "Error~"))
#endif
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

    typedef struct {
        PyObject_HEAD
        ObjId owner;
        char * name;
    } _Field;
    //////////////////////////////////////////
    // Methods for ElementField class
    //////////////////////////////////////////
    static int moose_ElementField_setNum(_Field * self, PyObject * num, void * closure);
    static PyObject* moose_ElementField_getNum(_Field * self, void * closure);
    static Py_ssize_t moose_ElementField_getLen(_Field * self, void * closure);
    static PyObject * moose_ElementField_getItem(_Field * self, Py_ssize_t index);
    static PyObject * moose_ElementField_getPath(_Field * self, void * closure);
    static PyObject * moose_ElementField_getId(_Field * self, void * closure);
    
    //////////////////////////////////////////
    // Methods for Id class
    //////////////////////////////////////////
    static int moose_Id_init(_Id * self, PyObject * args, PyObject * kwargs);
    static long moose_Id_hash(_Id * self);
    
    static PyObject * moose_Id_repr(_Id * self);
    static PyObject * moose_Id_str(_Id * self);
    static PyObject * moose_Id_delete(_Id * self);
    static PyObject * moose_Id_getValue(_Id * self);
    static PyObject * moose_Id_getPath(_Id * self);
    /* Id functions to allow part of sequence protocol */
    static Py_ssize_t moose_Id_getLength(_Id * self);
    static PyObject * moose_Id_getItem(_Id * self, Py_ssize_t index);
    static PyObject * moose_Id_getSlice(_Id * self, PyObject * args);    
    static PyObject * moose_Id_getShape(_Id * self);
    static PyObject * moose_Id_subscript(_Id * self, PyObject * op);
    static PyObject * moose_Id_richCompare(_Id * self, PyObject * args, int op);
    static int moose_Id_contains(_Id * self, PyObject * args);
    static PyObject * moose_Id_getattro(_Id * self, PyObject * attr);
    static int moose_Id_setattro(_Id * self, PyObject * attr, PyObject * value);
    static PyObject * moose_Id_setField(_Id * self, PyObject *args);
    ///////////////////////////////////////////
    // Methods for ObjId class
    ///////////////////////////////////////////
    static int moose_ObjId_init(PyObject * self, PyObject * args, PyObject * kwargs);
    static long moose_ObjId_hash(_ObjId * self);
    static PyObject * moose_ObjId_repr(_ObjId * self);
    static PyObject * moose_ObjId_getattro(_ObjId * self, PyObject * attr);
    static PyObject * moose_ObjId_getField(_ObjId * self, PyObject * args);
    static int moose_ObjId_setattro(_ObjId * self, PyObject * attr, PyObject * value);
    static PyObject * moose_ObjId_setField(_ObjId * self, PyObject * args);
    static PyObject * moose_ObjId_getLookupField(_ObjId * self, PyObject * args);
    static PyObject * moose_ObjId_setLookupField(_ObjId * self, PyObject * args);
    static PyObject * moose_ObjId_setDestField(_ObjId * self, PyObject * args);
    static PyObject * moose_ObjId_getFieldNames(_ObjId * self, PyObject * args);
    static PyObject * moose_ObjId_getFieldType(_ObjId * self, PyObject * args);
    static PyObject * moose_ObjId_getDataIndex(_ObjId * self);
    static PyObject * moose_ObjId_getFieldIndex(_ObjId * self);
    static PyObject * moose_ObjId_getNeighbors(_ObjId * self, PyObject * args);
    static PyObject * moose_ObjId_getId(_ObjId * self);
    static PyObject * moose_ObjId_connect(_ObjId * self, PyObject * args);
    static PyObject * moose_ObjId_richcompare(_ObjId * self, PyObject * args, int op);

    ////////////////////////////////////////////
    // Methods for LookupField
    ////////////////////////////////////////////
    static int moose_Field_init(_Field * self, PyObject * args, PyObject * kwds);
    static long moose_Field_hash(_Field * self);
    static PyObject * moose_Field_repr(_Field * self);
    static PyObject * moose_LookupField_getItem(_Field * self, PyObject * key);
    static int moose_LookupField_setItem(_Field * self, PyObject * key, PyObject * value);
    
    
    ////////////////////////////////////////////////
    // static functions to be accessed from Python
    ////////////////////////////////////////////////


    // The following are global functions
    static PyObject * oid_to_element(ObjId oid);
    static PyObject * moose_element(PyObject * dummy, PyObject * args);
    static PyObject * moose_useClock(PyObject * dummy, PyObject * args);
    static PyObject * moose_setClock(PyObject * dummy, PyObject * args);
    static PyObject * moose_start(PyObject * dummy, PyObject * args);
    static PyObject * moose_reinit(PyObject * dummy, PyObject * args);
    static PyObject * moose_stop(PyObject * dummy, PyObject * args);
    static PyObject * moose_isRunning(PyObject * dummy, PyObject * args);
    static PyObject * moose_exists(PyObject * dummy, PyObject * args);
    static PyObject * moose_loadModel(PyObject * dummy, PyObject * args);
    static PyObject * moose_saveModel(PyObject * dummy, PyObject * args);
    static PyObject * moose_writeSBML(PyObject * dummy, PyObject * args);
    static PyObject * moose_setCwe(PyObject * dummy, PyObject * args);
    static PyObject * moose_getCwe(PyObject * dummy, PyObject * args);
    static PyObject * moose_copy(PyObject * dummy, PyObject * args, PyObject * kwargs);
    static PyObject * moose_move(PyObject * dummy, PyObject * args);
    static PyObject * moose_delete(PyObject * dummy, PyObject * args);
    static PyObject * moose_connect(PyObject * dummy, PyObject * args);
    static PyObject * moose_getFieldDict(PyObject * dummy, PyObject * args);
    static PyObject * moose_getField(PyObject * dummy, PyObject * args);
    static PyObject * moose_syncDataHandler(PyObject * dummy, PyObject * target);
    static PyObject * moose_seed(PyObject * dummy, PyObject * args);
    static PyObject * moose_wildcardFind(PyObject * dummy, PyObject * args);
    // This should not be required or accessible to the user. Put here
    // for debugging threading issue.
    static PyObject * moose_quit(PyObject * dummy);
    
    //////////////////////////////////////////////////////////////
    // These are internal functions and not exposed in Python
    //////////////////////////////////////////////////////////////
    static PyObject * getLookupField(ObjId oid, char * fieldName, PyObject * key);
    static int setLookupField(ObjId oid, char * fieldName, PyObject * key, PyObject * value);
    static int define_class(PyObject * module_dict, const Cinfo * cinfo);
    static int define_destFinfos(const Cinfo * cinfo);
    static int defineAllClasses(PyObject* module_dict);
    static int define_lookupFinfos(const Cinfo * cinfo);
    static int define_elementFinfos(const Cinfo * cinfo);
    static PyObject * moose_ObjId_get_lookupField_attr(PyObject * self, void * closure);
    static PyObject * moose_ObjId_get_elementField_attr(PyObject * self, void * closure);
    static PyObject * moose_ObjId_get_destField_attr(PyObject * self, void * closure);
    static PyObject * _setDestField(ObjId oid, PyObject * args);
#if PY_MAJOR_VERSION >= 3
    PyMODINIT_FUNC PyInit_moose();
#else
    PyMODINIT_FUNC init_moose();
#endif


    int inner_getFieldDict(Id classId, string finfoType, vector<string>& fields, vector<string>& types); 

extern PyTypeObject ObjIdType;
extern PyTypeObject IdType;

    
} //!extern "C"


#define PYSEQUENCE_TO_VECTOR(BASETYPE, SEQUENCE){                       \
        Py_ssize_t length = PySequence_Length(SEQUENCE);            \
        vector < BASETYPE > * RET;                                       \
        BASETYPE * value;                                                \
        RET = new vector<BASETYPE>();                            \
        for (unsigned int ii = 0; ii < length; ++ii){               \
            PyObject * item = PySequence_GetItem(SEQUENCE, ii);     \
            if (item == NULL){                                      \
                ostringstream error;                                \
                error << "Item # " << ii << " is NULL";                 \
                PyErr_SetString(PyExc_ValueError, error.str().c_str());  \
            } else {                                                    \
                value = (BASETYPE*)to_cpp< BASETYPE >(item);         \
                if (value == NULL){                                     \
                    PyErr_SetString(PyExc_TypeError, "Cannot handle sequence of type "#BASETYPE); \
                } else {                                                \
                    RET->push_back(*value);                             \
                    delete value;                                       \
                }                                                       \
            }                                                           \
        }                                                               \
        return RET;                                                     \
    }

#define PYSEQUENCE_TO_VECVEC(BASETYPE, SEQUENCE){   \
        Py_ssize_t length1 = PySequence_Length(SEQUENCE);               \
        vector < vector <BASETYPE> > * RET = new vector < vector < BASETYPE > >((unsigned)length1); \
        for (unsigned int ii = 0; ii < length1; ++ii){                  \
            PyObject * subseq = PySequence_GetItem(SEQUENCE, ii);       \
            if (subseq == NULL){                                        \
                ostringstream error;                                    \
                error << "PYSEQUENCE_TO_VECVEC: Converting Python sequence of sequence to vector of vectors: Item # " \
                      << ii << " is NULL.";                             \
                PyErr_SetString(PyExc_ValueError, error.str().c_str()); \
                return RET;                                             \
            }                                                           \
            if (!PySequence_Check(subseq)){                             \
                PyErr_SetString(PyExc_TypeError, "PYSEQUENCE_TO_VECVEC: expected a sequence of sequences. Found oridinary sequence."); \
                return RET;                                             \
            }                                                           \
            Py_ssize_t length2 = PySequence_Length(subseq);             \
            for (unsigned jj = 0; jj < length2; ++jj){                  \
                PyObject * item = PySequence_GetItem(subseq, jj);       \
                if (item == NULL){                                      \
                    ostringstream error;                                \
                    error << "PYSEQUENCE_TO_VECVEC: found a null for "  \
                          << jj << "-th item on" << ii                  \
                          << "-th subsequence";                         \
                    PyErr_SetString(PyExc_ValueError, error.str().c_str()); \
                    return RET;                                         \
                }                                                       \
                BASETYPE * value = (BASETYPE*)to_cpp<BASETYPE>(item);   \
                RET->at(ii).push_back(*value);                          \
                delete value;                                           \
            }                                                           \
        }                                                               \
        return RET;                                                     \
    }

/// Converts PyObject to C++ object
/// Returns a pointer to the converted object. Deallocation is caller responsibility
template <class A> void * to_cpp(PyObject * object)
{
    if (typeid(A) == typeid(int)){
        int * ret = new int();
        * ret = PyInt_AsLong(object);
        return (void*)ret;
    } else if (typeid(A) == typeid(long)){
        long v = PyInt_AsLong(object);
        long * ret = new long();
        *ret = v;
        return (void*)ret;
    } else if (typeid(A) == typeid(short)){
        short v = PyInt_AsLong(object);
        short * ret = new short();
        *ret = v;
        return (void*)ret;
    } else if (typeid(A) == typeid(float)){
        float v = (float)PyFloat_AsDouble(object);
        if ( v == -1.0 && PyErr_Occurred()){
            PyErr_SetString(PyExc_TypeError, "Expected a sequence of floating point numbers.");
        } else {
            float * ret = new float();
            *ret = v;
            return (void*)ret;
        }
    } else if (typeid(A) == typeid(double)){
        double v = PyFloat_AsDouble(object);
        if ( v == -1.0 && PyErr_Occurred()){
            PyErr_SetString(PyExc_TypeError, "Expected a sequence of floating point numbers.");
        } else {
            double * ret = new double();
            *ret = v;
            return (void*)ret;
        }
    } else if (typeid(A) == typeid(string)) {
        char* tmp = PyString_AsString(object);
        if (tmp == NULL){
            return NULL;
        }        
        string * ret = new string(tmp);
        return (void*)ret;
    } else if (typeid(A) == typeid(unsigned int)) {
        unsigned int v = PyInt_AsUnsignedLongMask(object);
        unsigned int * ret = new unsigned int();
        *ret = v;
        return (void*)ret;
    } else if (typeid(A) == typeid(unsigned long)) {
        unsigned long v = PyInt_AsUnsignedLongMask(object);
        unsigned long * ret = new unsigned long();
        *ret = v;
        return (void*)ret;
    } else if (typeid(A) == typeid(Id))
    {
        _Id * value = (_Id*)object;
        if (value != NULL){
            Id * ret = new Id();
            * ret = value->id_;
        return (void*)ret;
        }
    } else if (typeid(A) == typeid(ObjId)){
        _ObjId * value = (_ObjId*)object;
        if (value != NULL){
            ObjId * ret = new ObjId();
            * ret = value->oid_;
            return (void*)ret;
        }
    } else if (typeid(A) == typeid(vector<int>)){
        PYSEQUENCE_TO_VECTOR( int, object)
    } else if (typeid(A) == typeid(vector<unsigned int>)){
        PYSEQUENCE_TO_VECTOR( unsigned int, object)
    } else if (typeid(A) == typeid(vector<short>)){
        PYSEQUENCE_TO_VECTOR( short, object)
    } else if (typeid(A) == typeid(vector<long>)){
        PYSEQUENCE_TO_VECTOR( long, object)
    } else if (typeid(A) == typeid(vector<unsigned long>)){
        PYSEQUENCE_TO_VECTOR( unsigned long, object)
    } else if (typeid(A) == typeid(vector<float>)){
        PYSEQUENCE_TO_VECTOR( float, object)
    } else if (typeid(A) == typeid(vector<double>)){
        PYSEQUENCE_TO_VECTOR( double, object)
    } else if (typeid(A) == typeid(vector<string>)){
        PYSEQUENCE_TO_VECTOR( string, object)
    } else if (typeid(A) == typeid(vector<ObjId>)){
        PYSEQUENCE_TO_VECTOR( ObjId, object)        
    } else if (typeid(A) == typeid(vector<Id>)){
        PYSEQUENCE_TO_VECTOR( Id, object)
    } else if (typeid(A) == typeid(vector< vector <double> >)){
        PYSEQUENCE_TO_VECVEC(double, object);
    } else if (typeid(A) == typeid(vector <vector <int > >)){
        PYSEQUENCE_TO_VECVEC(int, object);
    } else if (typeid(A) == typeid(vector < vector < unsigned int> >)){
        PYSEQUENCE_TO_VECVEC(unsigned int, object);
    }
    return NULL;
}

/// Set a destinfo that takes a vector argument
template <class A> inline PyObject* _set_vector_destFinfo(ObjId obj, string fieldName, int argIndex, PyObject * value)
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
    vector<A> * _value = (vector <A> *)to_cpp < vector <A> >(value);
    if (_value == NULL){
        return NULL;
    }
    bool ret = SetGet1< vector < A > >::set(obj, fieldName, *_value);
    delete _value;
    if (ret){
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}
            // assert(value_ptr) != NULL);
            // vector<Id> value = LookupField< KeyType, vector<Id> > ::get(oid, fname, *cpp_key);
            // for (unsigned ii = 0; ii < value.length(); ++ii){
            //     PyObject* item = PyObject_New(_Id, &IdType);
            //     ((_Id*)item)->id_ = item;
            //     if (PyList_Append(value_ptr, item) < 0){
            //         Py_XDECREF(item);
            //         Py_XDECREF(value_ptr);
            //         return NULL;
            //     }
            // }
            // ret = (PyObject*)value_ptr;
            // break;


/**
   Convert simple C++ data `v` of type A into Python object.
 */
template <typename A> PyObject *
to_py(A* v)
{    
    if (typeid(A) == typeid(bool)){
        if (*((bool*)v)){
            Py_RETURN_TRUE;
        } else {
            Py_RETURN_FALSE;
        }        
    } else if (typeid(A) == typeid(int)){
        return PyInt_FromLong(*((int*)v));
    } else if ( typeid(A) == typeid(short)){
        return PyInt_FromLong(*((short*)v));
    }else if (typeid(A) == typeid(long)){
        return PyLong_FromLong(*((long*)v));
    } else if (typeid(A) == typeid(float)){
        return PyFloat_FromDouble(*((float*)v));
    } else if (typeid(A) == typeid(double)){
        return PyFloat_FromDouble(*((double*)v));
    } else if (typeid(A) == typeid(string)){
        return PyString_FromString((*((string*)v)).c_str());
    } else if (typeid(A) == typeid(unsigned int)){
        return PyLong_FromUnsignedLong(*((unsigned int*)v));
    } else if (typeid(A) == typeid(unsigned long)){
        return PyLong_FromUnsignedLong(*((unsigned long*)v));
    }
#ifdef HAVE_LONG_LONG            
    else if (typeid(A) == typeid(long long)) {
        return  PyLong_FromLongLong(*((long long *)v));
    } else if (typeid(A) == typeid(unsigned long long)){                       
        return PyLong_FromUnsignedLongLong(*((unsigned long long*)v));
    }
#endif    
    else if (typeid(A) == typeid(ObjId)){
        _ObjId * obj = PyObject_New(_ObjId, &ObjIdType);
        obj->oid_ = *((ObjId*)v);
        return (PyObject*)obj;
    } else if (typeid(A) == typeid(Id)){
        _Id * obj = PyObject_New(_Id, &IdType);
        obj->id_ = *((Id*)v);
        return (PyObject*)obj;
    }
    return NULL;
}

/**
   Convert C++ vector `value` into a Python list.
 */
template<typename T> PyObject * to_pylist(vector<T>& value, PyObject * ret)
{
    assert(ret!=NULL);
    assert(PyList_Check(ret));
    assert(PyList_Size(ret) == 0);
    for (unsigned int ii = 0; ii < value.size(); ++ii){
        PyObject * v = to_py<T>(&value[ii]);
        if (PyList_Append(ret, v) < 0){
            Py_XDECREF(v);
            return NULL;
        }
    }
    return ret;    
}

/**
   Return vector value for `key` from LookupField `fieldname` of ObjId
   `oid`.
   
   Side effect: ret is populated with Python objects for individual
   entries in the vector value.
   
   Returns NULL on failure, `ret` otherwise.  */
template <typename KeyType, typename ValueType>
PyObject * get_vec_lookupfield(ObjId oid, string fieldname, KeyType key, PyObject * ret)
{
    assert(ret!=NULL);
    assert(PyList_Check(ret));
    assert(PyList_Size(ret) == 0);
    vector<ValueType> value = LookupField<KeyType, vector<ValueType> >::get(oid, fieldname, key);
    return to_pylist(value, ret);
}

/**
   Lookup entry for `key` from lookup field `fieldname` of object
   `oid`.  Here the lookup field must have primitive data types or
   ObjId or Id as the value. No vectors or other fancy datastructures
   are handled.  */
template <typename KeyType, typename ValueType>
PyObject * get_simple_lookupfield(ObjId oid, string fieldname, KeyType key)
{
    ValueType value = LookupField<KeyType, ValueType>::get(oid, fieldname, key);
    PyObject * v = to_py<ValueType>(&value);
    return v;
}

/**
   Get the value for key from LookupField fname.
 */
// The circus with value_ptr is to allow Id, ObjId and vectors to be
// allocated.
template <class KeyType> inline PyObject *
lookup_value(const ObjId& oid,
             string fname,
             char value_type_code,
             char key_type_code,
             PyObject * key,
             void * value_ptr=NULL)
{
    PyObject * ret = NULL;
    KeyType * cpp_key = (KeyType *)to_cpp<KeyType>(key);
    if (cpp_key == NULL){
        return NULL;
    }
    string value_type_str = string(1, value_type_code);
    switch (value_type_code){
        case 'b': // boolean is a special case that PyBuildValue does not handle
            ret = get_simple_lookupfield < KeyType, bool >(oid, fname, *cpp_key);
            break;
        case 'c': 
            ret = get_simple_lookupfield < KeyType, char >(oid, fname, *cpp_key);
            break;
        case 'h': 
            ret = get_simple_lookupfield < KeyType, short >(oid, fname, *cpp_key);
            break;
        case 'H': 
            ret = get_simple_lookupfield< KeyType, unsigned short >(oid, fname, *cpp_key);
            break;
        case 'i': 
            ret = get_simple_lookupfield < KeyType, int >(oid, fname, *cpp_key);
            break;
        case 'I': 
            ret = get_simple_lookupfield < KeyType, unsigned int >(oid, fname, *cpp_key);
            break;
        case 'l': 
            ret = get_simple_lookupfield< KeyType, long >(oid, fname, *cpp_key);
            break;
        case 'k': 
            ret = get_simple_lookupfield< KeyType, unsigned long >(oid, fname, *cpp_key);
            break;
#ifdef HAVE_LONG_LONG            
        case 'L': 
            ret = get_simple_lookupfield< KeyType, long long>(oid, fname, *cpp_key);
            break;
        case 'K': 
            ret = get_simple_lookupfield< KeyType, unsigned long long>(oid, fname, *cpp_key);
            break;
#endif
        case 'd':
            ret = get_simple_lookupfield< KeyType, double>(oid, fname, *cpp_key);
            break;
        case 'f': 
            ret = get_simple_lookupfield< KeyType, float >(oid, fname, *cpp_key);
            break;
        case 'x':
            ret = get_simple_lookupfield<KeyType, Id>(oid, fname, *cpp_key);
            break;
        case 'y':
            ret = get_simple_lookupfield< KeyType, Id >(oid, fname, *cpp_key);
            break;
        case 'X': 
            ret = get_vec_lookupfield<KeyType, Id>(oid, fname, *cpp_key, (PyObject*)value_ptr);
            break;
        case 'Y': 
            ret = get_vec_lookupfield<KeyType, ObjId>(oid, fname, *cpp_key, (PyObject*)value_ptr);
            break;
        case 'v': 
            ret = get_vec_lookupfield<KeyType, int>(oid, fname, *cpp_key, (PyObject*)value_ptr);
            break;
        case 'w': 
            ret = get_vec_lookupfield<KeyType, short>(oid, fname, *cpp_key, (PyObject*)value_ptr);
            break;
        case 'U': 
            ret = get_vec_lookupfield<KeyType, unsigned int>(oid, fname, *cpp_key, (PyObject*)value_ptr);
            break;
        case 'F': 
            ret = get_vec_lookupfield<KeyType, float>(oid, fname, *cpp_key, (PyObject*)value_ptr);
            break;
        case 'D': 
            ret = get_vec_lookupfield<KeyType, double>(oid, fname, *cpp_key, (PyObject*)value_ptr);
            break;
        case 'S': 
            ret = get_vec_lookupfield<KeyType, string>(oid, fname, *cpp_key, (PyObject*)value_ptr);
            break;
        default:
            PyErr_SetString(PyExc_TypeError, "invalid value type");
    }
    delete cpp_key;
    return ret;
}

/// The circus with value_ptr is to allow Id, ObjId to be allocated.
template <class KeyType> inline int set_lookup_value(const ObjId& oid, string fname, char value_type_code, char key_type_code, PyObject * key, PyObject * value_obj)
{
    bool success = false;
    KeyType *cpp_key = (KeyType*)to_cpp<KeyType>(key);
    if (cpp_key == NULL){
        return -1;
    }
#define SET_LOOKUP_VALUE( TYPE )                                        \
    {                                                                   \
        TYPE * value = (TYPE*)to_cpp<TYPE>(value_obj);                 \
            if (value){                                                 \
                success = LookupField < KeyType, TYPE > ::set(oid, fname, *cpp_key, *value); \
                delete value;                                           \
                delete cpp_key;                                         \
            }                                                           \
            break;                                                      \
    }
    
    string value_type_str = string(1, value_type_code);
    switch (value_type_code){
        case 'b':
            SET_LOOKUP_VALUE(bool)        
        case 'c':
            SET_LOOKUP_VALUE(char)
        case 'h':
            SET_LOOKUP_VALUE(short)
        case 'H':
            SET_LOOKUP_VALUE(unsigned short)
        case 'i':
            SET_LOOKUP_VALUE(int)
        case 'I':
            SET_LOOKUP_VALUE(unsigned int)
        case 'l':
            SET_LOOKUP_VALUE(long)
                    
        case 'k': 
            SET_LOOKUP_VALUE(unsigned long)
#ifdef HAVE_LONG_LONG
        case 'L':
            SET_LOOKUP_VALUE(long long)
        case 'K': 
            SET_LOOKUP_VALUE(unsigned long long);
#endif
        case 'd':
            SET_LOOKUP_VALUE(double)
        case 'f': 
            SET_LOOKUP_VALUE(float)
        case 's':
            SET_LOOKUP_VALUE(string)
        case 'x':
            SET_LOOKUP_VALUE(Id)
        case 'y':
            SET_LOOKUP_VALUE(ObjId)
        default:
            PyErr_SetString(PyExc_TypeError, "invalid value type");
    }
    if (success){
        return 0;
    } else {
        return -1;
    }
}

#endif // _MOOSEMODULE_H

// 
// moosemodule.h ends here
