// moosemodule.cpp --- 
// 
// Filename: moosemodule.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Thu Mar 10 11:26:00 2011 (+0530)
// Version: 
// Last-Updated: Tue Jul 23 20:27:30 2013 (+0530)
//           By: subha
//     Update #: 11006
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
//
// 2012-04-20 11:35:59 (+0530)
//
// This version will crash for debug build of Python. In Python the
// flag Py_TPFLAGS_HEAPTYPE is heavily overloaded and hinders dynamic
// class definition. Python expects non-heap-types to be defined
// statically. But we need to define the MOOSE classes dynamically by
// traversing the class definition objects (Cinfo) under /classes/
// element which are setup at initialization. Once defined, these
// class objects are not to be deallocated until program exit. Since
// all malloced memory is from heap, these classes qualify for
// heaptype, but setting Py_TPFLAGS_HEAPTYPE causes many other issues.
// One I encountered was that if HEAPTYPE is true, then
// help(classname) tries to convert the class object to a heaptype
// object (resulting in an invalid pointer) and causes a segmentation
// fault. If heaptype is not set it uses tp_name to print the help.
// See the following link for a discussion about this:
// http://mail.python.org/pipermail/python-dev/2009-July/090921.html
// 
// On the other hand, if we do not set Py_TPFLAGS_HEAPTYPE, GC tries
// tp_traverse on these classes (even when I unset Py_TPFLAGS_HAVE_GC)
// and fails the assertion in debug build of Python:
//
// python: Objects/typeobject.c:2683: type_traverse: Assertion `type->tp_flags & Py_TPFLAGS_HEAPTYPE' failed.
//
// Other projects have also encountered this issue:
//  https://bugs.launchpad.net/meliae/+bug/893461
// And from the comments it seems that the bug does not really hurt.
// 
// See also:
// http://stackoverflow.com/questions/8066438/how-to-dynamically-create-a-derived-type-in-the-python-c-api
// and the two discussions in Python mailing list referenced there.



// Change log:
// 
// 2011-03-10 Initial version. Starting coding directly with Python
//            API.  Trying out direct access to Python API instead of
//            going via SWIG. SWIG has this issue of creating huge
//            files and the resulting binaries are also very
//            large. Since we are not going to use any language but
//            Python in the foreseeable future, we can avoid the bloat
//            by coding directly with Python API.
//
// 2012-01-05 Much polished version. Handling destFinfos as methods in
//            Python class.
//
// 2012-04-13 Finished reimplementing the meta class system using
//            Python/C API.
//            Decided not to expose any lower level moose API.
//
// 2012-04-20 Finalized the C interface

// Code:

//////////////////////////// Headers ////////////////////////////////

#include <Python.h>
#include <structmember.h> // This defines the type id macros like T_STRING
#include "numpy/arrayobject.h"

#include <iostream>
#include <typeinfo>
#include <cstring>
#include <map>
#include <ctime>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "../basecode/header.h"
#include "../basecode/Id.h"
#include "../basecode/ObjId.h"
#include "../utility/utility.h"
#include "../randnum/randnum.h"
#include "../shell/Shell.h"

#include "moosemodule.h"

using namespace std;

//////////////////////// External functions /////////////////////////

extern void testSync();
extern void testAsync();
extern void testSyncArray( unsigned int size,
                           unsigned int numThreads,
                           unsigned int method );
extern void testShell();
extern void testScheduling();
extern void testSchedulingProcess();
extern void testBuiltins();
extern void testBuiltinsProcess();

extern void testMpiScheduling();
extern void testMpiBuiltins();
extern void testMpiShell();
extern void testMsg();
extern void testMpiMsg();
extern void testKinetics();
extern void nonMpiTests(Shell *);
extern void mpiTests();
extern void processTests( Shell* );
extern void test_moosemodule();
    


extern Id init(int argc, char ** argv, bool& doUnitTests, bool& doRegressionTests);

extern void initMsgManagers();
extern void destroyMsgManagers();
// extern void speedTestMultiNodeIntFireNetwork( 
//     unsigned int size, unsigned int runsteps );
// extern void regressionTests();
// extern bool benchmarkTests( int argc, char** argv );
// extern int getNumCores();


// C-wrapper to be used by Python
extern "C" {
    // IdType and ObjIdType are defined in ematrix.cpp and
    // melement.cpp respectively.
    extern PyTypeObject IdType;
    extern PyTypeObject ObjIdType;
    extern PyTypeObject moose_DestField;
    extern PyTypeObject moose_LookupField;
    extern PyTypeObject moose_ElementField;
    
    /////////////////////////////////////////////////////////////////
    // Module globals 
    /////////////////////////////////////////////////////////////////
    static int verbosity = 1;
    // static int isSingleThreaded = 0;
    static int isInfinite = 0;
    static unsigned int numNodes = 1;
    static unsigned int numCores = 1;
    static unsigned int myNode = 0;
    // static unsigned int numProcessThreads = 0;
    static int doUnitTests = 0;
    static int doRegressionTests = 0;
    static int quitFlag = 0;

    /**
       Return numpy typenum for specified type.
    */
    int get_npy_typenum(const type_info& ctype)
    {
        if (ctype == typeid(float)){
            return NPY_FLOAT;
        } else if (ctype == typeid(double)){
            return NPY_DOUBLE;
        } else if (ctype == typeid(int)){
            return NPY_INT;
        } else if (ctype == typeid(unsigned int)){
            return NPY_UINT;
        } else if (ctype == typeid(long)){
            return NPY_LONG;
        } else if (ctype == typeid(unsigned long)){
            return NPY_ULONG;
        } else if (ctype == typeid(short)){
            return NPY_SHORT;
        } else if (ctype == typeid(unsigned short)){
            return NPY_USHORT;
        } else if (ctype == typeid(char)){
            return NPY_CHAR;
        } else if (ctype == typeid(bool)){
            return NPY_BOOL;
        } else if (ctype == typeid(Id) || ctype == typeid(ObjId)){
            return NPY_OBJECT;
        } else {
            cerr << "Cannot handle type: " << ctype.name() << endl;
            return -1;
        }
    }


    /**
       Utility function to convert an Python integer or a sequence
       object into a vector of dimensions
    */
    vector<int> pysequence_to_dimvec(PyObject * dims)
    {
        vector <int> vec_dims;
        Py_ssize_t num_dims = 1;
        long dim_value = 1;
        if (dims){
            // First try to use it as a tuple of dimensions
            if (PySequence_Check(dims)){
                num_dims = PySequence_Length(dims);
                for (Py_ssize_t ii = 0; ii < num_dims; ++ ii){
                    PyObject* dim = PySequence_GetItem(dims, ii);
                    dim_value = PyInt_AsLong(dim);
                    Py_XDECREF(dim);                    
                    if ((dim_value == -1) && PyErr_Occurred()){
                        return vec_dims;
                    }
                    vec_dims.push_back((unsigned int)dim_value);
                }
            } else if (PyInt_Check(dims)){ // 1D array
                dim_value = PyInt_AsLong(dims);
                if (dim_value <= 0){
                    dim_value = 1;
                }
                vec_dims.push_back(dim_value);
            }
        } else {
            vec_dims.push_back(dim_value);
        }
        return vec_dims;
    }

    /**
       Convert Python object into C++ data structure. The data
       structure is allocated here and it is the responsibility of the
       caller to free that memory.
    */
    void * to_cpp(PyObject * object, char typecode)
    {
        switch(typecode){
            case 'i': {
                int * ret = new int();
                * ret = PyInt_AsLong(object);
                return (void*)ret;
            }
            case 'l': {
                long v = PyInt_AsLong(object);
                long * ret = new long();
                *ret = v;
                return (void*)ret;
            }
            case 'h': {
                short v = PyInt_AsLong(object);
                short * ret = new short();
                *ret = v;
                return (void*)ret;
            }
            case 'f': {
                float v = (float)PyFloat_AsDouble(object);
                if ( v == -1.0 && PyErr_Occurred()){
                    PyErr_SetString(PyExc_TypeError, "Expected a sequence of floating point numbers.");
                } else {
                    float * ret = new float();
                    *ret = v;
                    return (void*)ret;
                }
            }
            case 'd': {
                double v = PyFloat_AsDouble(object);
                if ( v == -1.0 && PyErr_Occurred()){
                    PyErr_SetString(PyExc_TypeError, "Expected a sequence of floating point numbers.");
                } else {
                    double * ret = new double();
                    *ret = v;
                    return (void*)ret;
                }
            }
            case 's': {
                char* tmp = PyString_AsString(object);
                if (tmp == NULL){
                    return NULL;
                }        
                string * ret = new string(tmp);
                return (void*)ret;
            }
            case 'I': {
                unsigned int v = PyInt_AsUnsignedLongMask(object);
                unsigned int * ret = new unsigned int();
                *ret = v;
                return (void*)ret;
            }
            case 'k': {
                unsigned long v = PyInt_AsUnsignedLongMask(object);
                unsigned long * ret = new unsigned long();
                *ret = v;
                return (void*)ret;
            }
            case 'x': {
                _Id * value = (_Id*)object;
                if (value != NULL){
                    Id * ret = new Id();
                    * ret = value->id_;
                    return (void*)ret;
                }
            }
            case 'y': {
                _ObjId * value = (_ObjId*)object;
                if (value != NULL){
                    ObjId * ret = new ObjId();
                    * ret = value->oid_;
                    return (void*)ret;
                }
            }
            case 'v':
                return PySequenceToVector< int >(object, 'i');
            case 'N':
                return PySequenceToVector < unsigned int > (object, 'I');
            case 'w':
                return PySequenceToVector < short > (object, 'h');
            case 'M':
                return PySequenceToVector < long > (object, 'l');
            case 'P':
                return PySequenceToVector < unsigned long > (object, 'k');
            case 'F':
                return PySequenceToVector < float > (object, 'f');
            case 'D':
                return PySequenceToVector < double > (object, 'd');
            case 'S':
                return PySequenceToVector < string > (object, 's');
            case 'Y':
                return PySequenceToVector < ObjId > (object, 'y');      
            case 'X':
                return PySequenceToVector < Id > (object, 'x');
            case 'R':
                return PySequenceToVectorOfVectors< double >(object, 'd');
            case 'Q':
                return PySequenceToVectorOfVectors< int >(object, 'i');
            case 'T':
                return PySequenceToVectorOfVectors< unsigned int > (object, 'I');
        }
        return NULL;
    }


    /**
       Utility function to convert C++ object into Python object.
    */
    PyObject * to_py(void * obj, char typecode)
    {
        switch(typecode){
            case 'd': {
                double * ptr = static_cast<double *>(obj);
                assert(ptr != NULL);
                return PyFloat_FromDouble(*ptr);
            }
            case 's': { // handle only C++ string and NOT C char array because static cast cannot differentiate the two
                string * ptr = static_cast< string * >(obj);
                assert (ptr != NULL);
                return PyString_FromString(ptr->c_str());
            }
            case 'x': {
                Id * ptr = static_cast< Id * >(obj);
                assert(ptr != NULL);
                _Id * id = PyObject_New(_Id, &IdType);
                id->id_ = *ptr;
                return (PyObject *)(id);
            }
            case 'y': {
                ObjId * ptr = static_cast< ObjId * >(obj);
                assert(ptr != NULL);
                _ObjId * oid = PyObject_New(_ObjId, &ObjIdType);
                oid->oid_ = *ptr;
                return (PyObject*)oid;
            }
            case 'l': {
                long * ptr = static_cast< long * >(obj);
                assert(ptr != NULL);
                return PyLong_FromLong(*ptr);
            }
            case 'k': {
                unsigned long * ptr = static_cast< unsigned long * >(obj);
                assert(ptr != NULL);
                return PyLong_FromUnsignedLong(*ptr);
            }
            case 'i': {// integer
                int * ptr = static_cast< int * >(obj);
                assert(ptr != NULL);
                return PyInt_FromLong(*ptr);
            }
            case 'I': { // unsigned int
                unsigned int * ptr = static_cast< unsigned int * >(obj);
                assert(ptr != NULL);
                return PyLong_FromUnsignedLong(*ptr);
            }
            case 'b': {//bool
                bool * ptr = static_cast< bool * >(obj);
                assert(ptr != NULL);
                if (*ptr){
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }
#ifdef HAVE_LONG_LONG                
            case 'L': { //long long
                long long * ptr = static_cast< long long * > (obj);
                assert(ptr != NULL);
                return PyLong_FromLongLong(*ptr);
            }
            case 'K': { // unsigned long long
                unsigned long long * ptr = static_cast< unsigned long long * >(obj);
                assert(ptr != NULL);
                return PyLong_FromUnsignedLongLong(*ptr);
            }
#endif // HAVE_LONG_LONG
            case 'f': { // float
                float * ptr = static_cast< float* >(obj);
                assert(ptr != NULL);
                return PyFloat_FromDouble(*ptr);
            }
            case 'c': { // char
                char * ptr = static_cast< char * >(obj);
                assert(ptr != NULL);
                return Py_BuildValue("c", *ptr);
            }
            case 'h': { //short
                short * ptr = static_cast< short* >(obj);
                assert(ptr != NULL);
                return Py_BuildValue("h", *ptr);
            }
            case 'H': { // unsigned short
                unsigned short * ptr = static_cast< unsigned short * >(obj);
                assert(ptr != NULL);
                return Py_BuildValue("H", *ptr);
            }
            case 'D': case 'v': case 'M': case 'X': case 'Y': case 'C': case 'w': case 'N': case 'P': case 'F': case 'S': case 'T': case 'Q': case 'R': {
                return to_pytuple(obj, innerType(typecode));
            }
            default: {
                PyErr_SetString(PyExc_TypeError, "unhandled data type");
                return NULL;
            }                
        } // switch(typecode)
    } // to_py

    /**
       Inner function to convert C++ object at vptr and set tuple
       entry ii to the created PyObject. typecode is passed to to_cpp
       for conversion.  If error occurs while setting tuple antry, it
       decrements the refcount of the tuple and returns NULL. Returns
       the tuple on success.
    */
    PyObject * convert_and_set_tuple_entry(PyObject * tuple, unsigned int index, void * vptr, char typecode)
    {
        PyObject * item = to_py(vptr, typecode);
        if (item == NULL){
            return NULL; // the error message would have been populated by to_cpp
        }
        if (PyTuple_SetItem(tuple, (Py_ssize_t)index, item) != 0){
            PyErr_SetString(PyExc_RuntimeError, "convert_and_set_tuple_entry: could not set tuple entry.");
            Py_XDECREF(tuple);
            return NULL;
        }
        return tuple;
    }
    
    /**
       Convert a C++ vector to Python tuple
    */
    PyObject * to_pytuple(void * obj, char typecode)
    {
        PyObject * ret;
        switch (typecode){
            case 'd': { // vector<double>
                vector< double > * vec = static_cast< vector < double >* >(obj);
                assert(vec != NULL);
                npy_intp size = (npy_intp)(vec->size());
                ret = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
                assert(ret != NULL);
                char * ptr = PyArray_BYTES((PyArrayObject*)ret);
                memcpy(ptr, &(*vec)[0], size * sizeof(double));
                return ret;
            }
            case 'i': { // vector<int>
                vector< int > * vec = static_cast< vector < int >* >(obj);
                assert(vec != NULL);
                npy_intp size = (npy_intp)(vec->size());
                ret = PyArray_SimpleNew(1, &size, NPY_INT);
                assert(ret != NULL);
                char * ptr = PyArray_BYTES((PyArrayObject*)ret);
                memcpy(ptr, &(*vec)[0], size * sizeof(int));
                return ret;
            }
            case 'I': { // vector<unsigned int>
                vector< int > * vec = static_cast< vector < int >* >(obj);
                assert(vec != NULL);
                npy_intp size = (npy_intp)(vec->size());
                ret = PyArray_SimpleNew(1, &size, NPY_UINT);
                assert(ret != NULL);
                char * ptr = PyArray_BYTES((PyArrayObject*)ret);
                memcpy(ptr, &(*vec)[0], size * sizeof(unsigned int));
                return ret;
            }
            case 'l': { // vector<long>
                vector< long > * vec = static_cast< vector < long >* >(obj);
                assert(vec != NULL);
                npy_intp size = (npy_intp)(vec->size());
                ret = PyArray_SimpleNew(1, &size, NPY_INT);
                assert(ret != NULL);
                char * ptr = PyArray_BYTES((PyArrayObject*)ret);
                memcpy(ptr, &(*vec)[0], size * sizeof(long));
                return ret;
            }
            case 'x': { // vector<Id>
                vector< Id > * vec = static_cast< vector < Id >* >(obj);
                assert(vec != NULL);
                ret = PyTuple_New((Py_ssize_t)vec->size());
                assert(ret != NULL);
                for (unsigned int ii = 0; ii < vec->size(); ++ii){
                    if (convert_and_set_tuple_entry(ret, ii, (void*)&vec->at(ii), typecode) == NULL){
                        return NULL;
                    }
                }
                break;
            }
            case 'y': { // vector<ObjId>
                vector< ObjId > * vec = static_cast< vector < ObjId >* >(obj);
                assert(vec != NULL);
                ret = PyTuple_New((Py_ssize_t)vec->size());
                assert(ret != NULL);
                for (unsigned int ii = 0; ii < vec->size(); ++ii){
                    if (convert_and_set_tuple_entry(ret, ii, (void*)&vec->at(ii), typecode) == NULL){
                        return NULL;
                    }
                }
                break;
            }
            case 'c': { // vector<char>
                vector< char > * vec = static_cast< vector < char >* >(obj);
                assert(vec != NULL);
                ret = PyTuple_New((Py_ssize_t)vec->size());
                assert(ret != NULL);
                for (unsigned int ii = 0; ii < vec->size(); ++ii){
                    if (convert_and_set_tuple_entry(ret, ii, (void*)&vec->at(ii), typecode) == NULL){
                        return NULL;
                    }
                }
                break;
            }
            case 'h': { // vector<short>
                vector< short > * vec = static_cast< vector < short >* >(obj);
                assert(vec != NULL);
                ret = PyTuple_New((Py_ssize_t)vec->size());
                assert(ret != NULL);
                for (unsigned int ii = 0; ii < vec->size(); ++ii){
                    if (convert_and_set_tuple_entry(ret, ii, (void*)&vec->at(ii), typecode) == NULL){
                        return NULL;
                    }
                }
                break;
            }
            case 'k': { // vector<unsigned long>
                vector< unsigned int > * vec = static_cast< vector < unsigned int >* >(obj);
                assert(vec != NULL);
                npy_intp size = (npy_intp)(vec->size());
                ret = PyArray_SimpleNew(1, &size, NPY_UINT);
                assert(ret != NULL);
                char * ptr = PyArray_BYTES((PyArrayObject*)ret);
                memcpy(ptr, &(*vec)[0], size * sizeof(unsigned int));
                return ret;
            }
            case 'L': { // vector<long long> - this is not used at present
                vector< long long> * vec = static_cast< vector < long long>* >(obj);
                assert(vec != NULL);
                npy_intp size = (npy_intp)(vec->size());
                ret = PyArray_SimpleNew(1, &size, NPY_LONGLONG);
                assert(ret != NULL);
                char * ptr = PyArray_BYTES((PyArrayObject*)ret);
                memcpy(ptr, &(*vec)[0], size * sizeof(long long));
                return ret;
            }
            case 'K': { // vector<unsigned long long> - this is not used at present
                vector< unsigned long long> * vec = static_cast< vector < unsigned long long>* >(obj);
                assert(vec != NULL);
                npy_intp size = (npy_intp)(vec->size());
                ret = PyArray_SimpleNew(1, &size, NPY_ULONGLONG);
                assert(ret != NULL);
                char * ptr = PyArray_BYTES((PyArrayObject*)ret);
                memcpy(ptr, &(*vec)[0], size * sizeof(unsigned long long));
                return ret;
            }
            case 'F': { // vector<float>
                vector< float > * vec = static_cast< vector < float >* >(obj);
                assert(vec != NULL);
                npy_intp size = (npy_intp)(vec->size());
                ret = PyArray_SimpleNew(1, &size, NPY_FLOAT);
                assert(ret != NULL);
                char * ptr = PyArray_BYTES((PyArrayObject*)ret);
                memcpy(ptr, &(*vec)[0], size * sizeof(float));
                return ret;
            }
            case 's': { // vector<string>
                vector< string > * vec = static_cast< vector < string >* >(obj);
                assert(vec != NULL);
                ret = PyTuple_New((Py_ssize_t)vec->size());
                assert(ret != NULL);
                for (unsigned int ii = 0; ii < vec->size(); ++ii){
                    string v = vec->at(ii);
                    if (convert_and_set_tuple_entry(ret, ii, (void*)&v, typecode) == NULL){
                        return NULL;
                    }
                }
                break;
            }
            case 'N': { // vector< vector <unsigned int > >
                vector< vector< unsigned int > > * vec = static_cast< vector < vector< unsigned int > >* >(obj);
                assert(vec != NULL);
                ret = PyTuple_New((Py_ssize_t)vec->size());
                assert(ret != NULL);
                for (unsigned int ii = 0; ii < vec->size(); ++ii){
                    if (convert_and_set_tuple_entry(ret, ii, (void*)&vec->at(ii), typecode) == NULL){
                        return NULL;
                    }
                }
                break;
            }
            case 'v': { // vector< vector < int > >
                vector< vector< int > > * vec = static_cast< vector < vector< int > >* >(obj);
                assert(vec != NULL);
                ret = PyTuple_New((Py_ssize_t)vec->size());
                assert(ret != NULL);
                for (unsigned int ii = 0; ii < vec->size(); ++ii){
                    if (convert_and_set_tuple_entry(ret, ii, (void*)&vec->at(ii), typecode) == NULL){
                        return NULL;
                    }
                }
                break;
            }
            case 'D': { // vector< vector <double > >
                vector< vector< double > > * vec = static_cast< vector < vector< double > >* >(obj);
                assert(vec != NULL);
                ret = PyTuple_New((Py_ssize_t)vec->size());
                assert(ret != NULL);
                for (unsigned int ii = 0; ii < vec->size(); ++ii){
                    if (convert_and_set_tuple_entry(ret, ii, (void*)&vec->at(ii), typecode) == NULL){
                        return NULL;
                    }
                }
                break;
            }
            default:
                PyErr_SetString(PyExc_TypeError, "unhandled type");
                return NULL;
        }
            return ret;
    }
    
    // Global store of defined MOOSE classes.
    map<string, PyTypeObject *>& get_moose_classes()
    {
        static map<string, PyTypeObject *> defined_classes;
        return defined_classes;
    }
    
    // Global storage for PyGetSetDef structs for LookupFields.
    map<string, vector <PyGetSetDef> >& get_getsetdefs()
    {
        static map<string, vector <PyGetSetDef>  > getset_defs;
        return getset_defs;
    }
    
    // Global storage for every LookupField we create.
    map<string, PyObject *>& get_inited_lookupfields()
    {
        static map<string, PyObject *> inited_lookupfields;
        return inited_lookupfields;
    }
    
    map< string, PyObject * >& get_inited_destfields()
    {
        static map<string, PyObject * > inited_destfields;
        return inited_destfields;
    }

    map< string, PyObject *>& get_inited_elementfields()
    {
        static map< string, PyObject *> inited_elementfields;
        return inited_elementfields;
    }
    
    /**
       map of fields which are aliased in Python to avoid collision
       with Python keywords.
    */
    const map<string, string>& get_field_alias()
    {
        static map<string, string> alias;
        if (alias.empty()){
            // alias["class_"] = "class";
            alias["lambda_"] = "lambda";
        }
        return alias;
    }


    // Minimum number of arguments for setting destFinfo - 1-st
    // the finfo name.
    // Py_ssize_t minArgs = 1;
    
    // // Arbitrarily setting maximum on variable argument list. Read:
    // // http://www.swig.org/Doc1.3/Varargs.html to understand why
    // Py_ssize_t maxArgs = 10;

    
    ///////////////////////////////////////////////////////////////////////////
    // Helper routines
    ///////////////////////////////////////////////////////////////////////////

    
    /**
       Get the environment variables and set runtime environment based
       on them.
    */
    vector <string> setup_runtime_env()
    {
        const map<string, string>& argmap = getArgMap();
        vector<string> args;
        args.push_back("moose");
        map<string, string>::const_iterator it;
        // it = argmap.find("SINGLETHREADED");
        // if (it != argmap.end()){
        //     istringstream(it->second) >> isSingleThreaded;
        //     if (isSingleThreaded){
        //         args.push_back("-s");
        //     }
        // }
        it = argmap.find("INFINITE");
        if (it != argmap.end()){
            istringstream(it->second) >> isInfinite;
            if (isInfinite){
                args.push_back("-i");
            }
        }
        it = argmap.find("NUMNODES");
        if (it != argmap.end()){
            istringstream(it->second) >> numNodes;
            args.push_back("-n");
            args.push_back(it->second);            
        }
        it = argmap.find("NUMCORES");
        if (it != argmap.end()){
            istringstream(it->second) >> numCores;
        }
        // it = argmap.find("NUMPTHREADS");
        // if (it != argmap.end()){
        //     istringstream(it->second) >> numProcessThreads;
        //     args.push_back("-t");
        //     args.push_back(it->second);	            
        // }
        it = argmap.find("QUIT");
        if (it != argmap.end()){
            istringstream(it->second) >> quitFlag;
            if (quitFlag){
                args.push_back("-q");
            }
        }
        it = argmap.find("VERBOSITY");
        if (it != argmap.end()){
            istringstream(it->second) >> verbosity;            
        }
        // it = argmap.find("DOUNITTESTS");
        // if (it != argmap.end()){
        //     istringstream(it->second) >> doUnitTests;            
        // }
        // it = argmap.find("DOREGRESSIONTESTS");
        // if (it != argmap.end()){
        //     istringstream(it->second) >> doRegressionTests;            
        // }
        
        if (verbosity > 0){
            cout << "ENVIRONMENT: " << endl
                 << "----------------------------------------" << endl
                 // << "   SINGLETHREADED = " << isSingleThreaded << endl
                 << "   INFINITE = " << isInfinite << endl
                 << "   NUMNODES = " << numNodes << endl
                 // << "   NUMPTHREADS = " << numProcessThreads << endl
                 << "   VERBOSITY = " << verbosity << endl
                 // << "   DOUNITTESTS = " << doUnitTests << endl
                 // << "   DOREGRESSIONTESTS = " << doRegressionTests << endl
                 << "========================================" << endl;
        }
        return args;
    } //! setup_runtime_env()

    /**
       Create the shell instance unless already created. This calls
       basecode/main.cpp:init(argc, argv) to do the initialization.

       Return the Id of the Shell object.
    */
    Id get_shell(int argc, char ** argv)
    {
        static int inited = 0;
        if (inited){
            return Id(0);
        }
        bool dounit = doUnitTests != 0;
        bool doregress = doRegressionTests != 0;
        // Utilize the main::init function which has friend access to Id
        Id shellId = init(argc, argv, dounit, doregress);
        inited = 1;
        Shell * shellPtr = reinterpret_cast<Shell*>(shellId.eref().data());
        // if (dounit){
        //     nonMpiTests( shellPtr ); // These tests do not need the process loop.
        // }

        // if (!shellPtr->isSingleThreaded()){
        //     shellPtr->launchThreads();
        // }
        // if ( shellPtr->myNode() == 0 ) {
        //     if (dounit){
        //         mpiTests();
        //         processTests( shellPtr );
        //         regressionTests();
        //     }
        //     if ( benchmarkTests( argc, argv ) || quitFlag ){
        //         shellPtr->doQuit();
        //     }
        // }
        return shellId;
    } //! create_shell()

    /**
       Clean up after yourself.
    */
    void finalize()
    {
        static bool finalized = false;
        if (finalized){
            return;
        }
        finalized = true;
        Id shellId = get_shell(0, NULL);
        for (map<string, PyObject *>::iterator it =
                     get_inited_lookupfields().begin();
             it != get_inited_lookupfields().end();
             ++it){
            Py_XDECREF(it->second);
        }
        // Clear the memory for PyGetSetDefs. The key
        // (name) was dynamically allocated using calloc. So was the
        // docstring.
        for (map<string, vector<PyGetSetDef> >::iterator it =
                     get_getsetdefs().begin();
             it != get_getsetdefs().end();
             ++it){
            vector <PyGetSetDef> &getsets = it->second;
            for (unsigned int ii = 0; ii < getsets.size()-1; ++ii){ // the -1 is for the empty sentinel entry
                free(getsets[ii].name);
                Py_XDECREF(getsets[ii].closure);
            }
        }
        get_getsetdefs().clear();
        for (map<string, PyObject *>::iterator it = get_inited_destfields().begin();
             it != get_inited_destfields().end();
             ++it){
            Py_XDECREF(it->second);
        }
        SHELLPTR->doQuit();
        // Destroy the Shell object
        Neutral* ns = reinterpret_cast<Neutral*>(shellId.element()->data(0));
        ns->destroy( shellId.eref(), 0);
#ifdef USE_MPI
        MPI_Finalize();
#endif
    } //! finalize()


    /**
       Return list of available Finfo types.
       Place holder for static const to avoid static initialization issues.
    */
    const char ** getFinfoTypes()
    {
        static const char * finfoTypes[] = {"valueFinfo",
                                            "srcFinfo",
                                            "destFinfo",
                                            "lookupFinfo",
                                            "sharedFinfo",
                                            "fieldElementFinfo",
                                            0};
        return finfoTypes;
    }

    /**
       get the field type for specified field
    
       Argument:
       className -- class to look in
       
       fieldName -- field to look for
       
       finfoType -- finfo type to look in (can be valueFinfo,
       destFinfo, srcFinfo, lookupFinfo etc.
       
       Return:
       
       string -- value of type field of the Finfo object. This is a
       comma separated list of C++ template arguments
    */
    string getFieldType(string className, string fieldName, string finfoType)
    {
        string fieldType = "";
        string classInfoPath("/classes/" + className);
        Id classId(classInfoPath);
        assert (classId != Id());
        // unsigned int numFinfos = Field<unsigned int>::get(ObjId(classId, 0), "num_" + finfoType);
        Id fieldId(classId.path() + "/" + finfoType);
        for (unsigned int ii = 0; ii < Field<unsigned int>::get(fieldId, "numData"); ++ii){
            string _fieldName = Field<string>::get(ObjId(fieldId, ii, 0), "name");
            if (fieldName == _fieldName){                
                fieldType = Field<string>::get(ObjId(fieldId, ii, 0), "type");
                break;
            }
        }
        return fieldType;        
    }

    /**
       Parse the type field of Finfo objects.

       The types field is a comma separated list of the template
       arguments. We populate `typeVec` with the individual
       type strings.
    */
    int parse_Finfo_type(string className, string finfoType, string fieldName, vector<string> & typeVec)
    {
        string typestring = getFieldType(className, fieldName, finfoType);
        if (typestring.empty()){
            return -1;
        }
        tokenize(typestring, ",", typeVec);
        if ((int)typeVec.size() > maxArgs){
            return -1;
        }
        for (unsigned int ii = 0; ii < typeVec.size() ; ++ii){
            char type_code = shortType(typeVec[ii]);
            if (type_code == 0){
                return -1;
            }
        }
        return 0;
    }

    /**
       Return a pair containing (field name, finfo type). 
    */
    pair < string, string > getFieldFinfoTypePair(string className, string fieldName)
    {
        for (const char ** finfoType = getFinfoTypes(); *finfoType; ++finfoType){
            string ftype = getFieldType(className, fieldName, string(*finfoType));
            if (!ftype.empty()) {
                return pair < string, string > (ftype, string(*finfoType));
            }
        }
        return pair <string, string>("", "");
    }

    /**
       Return a vector of field names of specified finfo type.
    */
    vector<string> getFieldNames(string className, string finfoType)
    {
        vector <string> ret;
        Id classId("/classes/" + className);
        assert(classId != Id());
        unsigned int numFinfos = Field<unsigned int>::get(ObjId(classId), "num_" + finfoType);
        Id fieldId(classId.path() + "/" + finfoType);
        if (fieldId == Id()){
            return ret;
        }
        for (unsigned int ii = 0; ii < numFinfos; ++ii){
            string fieldName = Field<string>::get(ObjId(fieldId, DataId(0, ii, 0)), "name");
            ret.push_back(fieldName);
        }
        return ret;
    }

    /**
       Populate the `fieldNames` vector with names of the fields of
       `finfoType` in the specified class.

       Populate the `fieldTypes` vector with corresponding C++ data
       type string (Finfo.type).
    */
    int getFieldDict(Id classId, string finfoType,
                     vector<string>& fieldNames, vector<string>&fieldTypes)
    {
        unsigned int numFinfos =
                Field<unsigned int>::get(ObjId(classId),
                                         "num_" + string(finfoType));
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

    //////////////////////////
    // Field Type definition
    //////////////////////////
    /////////////////////////////////////////////////////
    // ObjId functions.
    /////////////////////////////////////////////////////


    /**
       Utility function to traverse python class hierarchy to reach closest base class.
       Ideally we should go via mro
    */
 string get_baseclass_name(PyObject * self)
    {
        extern PyTypeObject ObjIdType;
        string basetype_str = "";
        PyTypeObject * base = NULL;
        for (base = Py_TYPE(self);
             base != &ObjIdType; base = base->tp_base){
            basetype_str = base->tp_name;
            size_t dot = basetype_str.find('.');
            basetype_str = basetype_str.substr(dot+1);
            if (get_moose_classes().find(basetype_str) !=
                get_moose_classes().end()){
                return basetype_str;
            }
        }
        if (base == Py_TYPE(self)){
            return "Neutral";
        }
        return basetype_str;
    }
    
    ////////////////////////////////////////////
    // Module functions
    ////////////////////////////////////////////
    PyDoc_STRVAR(moose_getFieldNames_documentation,
                 "getFieldNames(className, finfoType='valueFinfo') -> tuple\n"
                 "\n"
                 "Get a tuple containing the name of all the fields of `finfoType`\n"
                 "kind.\n"
                 "\n"
                 "Parameters\n"
                 "----------\n"
                 "className : string\n"
                 "\tName of the class to look up.\n"
                 "finfoType : string\n"
                 "\tThe kind of field (`valueFinfo`, `srcFinfo`, `destFinfo`,\n"
                 "`lookupFinfo`, `fieldElementFinfo`.).\n");

    PyObject * moose_getFieldNames(PyObject * dummy, PyObject * args)
    {
        char * className = NULL;
        char _finfoType[] = "valueFinfo";
        char * finfoType = _finfoType;
        if (!PyArg_ParseTuple(args, "s|s", &className, &finfoType)){
            return NULL;
        }
        vector <string> fieldNames = getFieldNames(className, finfoType);
        PyObject * ret = PyTuple_New(fieldNames.size());
                
        for (unsigned int ii = 0; ii < fieldNames.size(); ++ii){
            if (PyTuple_SetItem(ret, ii, PyString_FromString(fieldNames[ii].c_str())) == -1){
                Py_XDECREF(ret);
                return NULL;
            }
        }
        return ret;
    }
    
    PyDoc_STRVAR(moose_copy_documentation,
                 "copy(src, dest, name, n, toGlobal, copyExtMsg) -> bool\n"
                 "Make copies of a moose object.\n"
                 "Parameters\n"
                 "----------\n"
                 "src : ematrix, element or str\n"
                 "\tsource object.\n"
                 "dest : ematrix, element or str\n"
                 "\tDestination object to copy into.\n"
                 "name : str\n"
                 "\tName of the new object. If omitted, name of the original will be used.\n"
                 "n : int\n"
                 "\tNumber of copies to make.\n"
                 "toGlobal: int\n"
                 "\tRelevant for parallel environments only. If false, the copies will\n"
                 "reside on local node, otherwise all nodes get the copies.\n"
                 "copyExtMsg: int\n"
                 "\tIf true, messages to/from external objects are also copied.\n"
                 "\n"
                 "Returns\n"
                 "-------\n"
                 "ematrix of the copied object\n"
                 );
    PyObject * moose_copy(PyObject * dummy, PyObject * args, PyObject * kwargs)
    {
        PyObject * src = NULL, * dest = NULL;
        char * newName = NULL;
        static const char * kwlist[] = {"src", "dest", "name", "n", "toGlobal", "copyExtMsg", NULL};
        unsigned int num=1, toGlobal=0, copyExtMsgs=0;
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|sIII", const_cast<char**>(kwlist), &src, &dest, &newName, &num, &toGlobal, &copyExtMsgs)){
            return NULL;
        }
        Id _src, _dest;
        if (PyObject_IsInstance(src, (PyObject*)&IdType)){
            _src = ((_Id*)src)->id_;
        } else if (PyObject_IsInstance(src, (PyObject*)&ObjIdType)){
            _src = ((_ObjId*)src)->oid_.id;
        } else if (PyString_Check(src)){
            _src = Id(PyString_AsString(src));
        } else {
            PyErr_SetString(PyExc_TypeError, "Source must be instance of ematrix, element or string.");
            return NULL;
        }
        if (_src == Id()){
            PyErr_SetString(PyExc_ValueError, "Cannot make copy of moose shell.");
            return NULL;
        } 
        if (PyObject_IsInstance(dest, (PyObject*)&IdType)){
            _dest = ((_Id*)dest)->id_;
        } else if (PyObject_IsInstance(dest, (PyObject*)&ObjIdType)){
            _dest = ((_ObjId*)dest)->oid_.id;
        } else if (PyString_Check(dest)){
            _dest = Id(PyString_AsString(dest));
        } else {
            PyErr_SetString(PyExc_TypeError, "destination must be instance of ematrix, element or string.");
            return NULL;
        }
        if (!Id::isValid(_src) || !Id::isValid(_dest)){
            RAISE_INVALID_ID(NULL, "moose_copy");
        }
        string name;
        if (newName == NULL){
            // Use the original name if name is not specified.
            name = Field<string>::get(ObjId(_src, 0), "name");
        } else {
            name = string(newName);
        }
        _Id * tgt = PyObject_New(_Id, &IdType);
        tgt->id_ = SHELLPTR->doCopy(_src, _dest, name, num, toGlobal, copyExtMsgs);
        PyObject * ret = (PyObject*)tgt;
        return ret;
    }

    // Not sure what this function should return... ideally the Id of the
    // moved object - does it change though?
    PyObject * moose_move(PyObject * dummy, PyObject * args)
    {
        PyObject * src, * dest;
        if (!PyArg_ParseTuple(args, "OO:moose_move", &src, &dest)){
            return NULL;
        }
        if (((_Id*)src)->id_ == Id()){
            PyErr_SetString(PyExc_ValueError, "cannot move moose shell");
            return NULL;
        }
        SHELLPTR->doMove(((_Id*)src)->id_, ((_Id*)dest)->id_);
        Py_RETURN_NONE;
    }

    PyDoc_STRVAR(moose_delete_documentation,
                 "moose.delete(id)"
                 "\n"
                 "\nDelete the underlying moose object. This does not delete any of the"
                 "\nPython objects referring to this ematrix but does invalidate them. Any"
                 "\nattempt to access them will raise a ValueError."
                 "\n"
                 "\nParameters\n"
                 "\n----------"
                 "\nid : ematrix"
                 "\n\tematrix of the object to be deleted."
                 "\n");
    PyObject * moose_delete(PyObject * dummy, PyObject * args)
    {
        PyObject * obj;
        if (!PyArg_ParseTuple(args, "O:moose.delete", &obj)){
            return NULL;
        }
        if (!PyObject_IsInstance(obj, (PyObject*)&IdType)){
            PyErr_SetString(PyExc_TypeError, "ematrix instance expected");
            return NULL;
        }
        if (((_Id*)obj)->id_ == Id()){
            PyErr_SetString(PyExc_ValueError, "cannot delete moose shell.");
            return NULL;
        }
        if (!Id::isValid(((_Id*)obj)->id_)){
            RAISE_INVALID_ID(NULL, "moose_delete");
        }
        deleteId((_Id*)obj);
        // SHELLPTR->doDelete(((_Id*)obj)->id_);
        Py_RETURN_NONE;
    }

    PyObject * moose_useClock(PyObject * dummy, PyObject * args)
    {
        char * path, * field;
        unsigned int tick;
        if(!PyArg_ParseTuple(args, "Iss:moose_useClock", &tick, &path, &field)){
            return NULL;
        }
        SHELLPTR->doUseClock(string(path), string(field), tick);
        Py_RETURN_NONE;
    }
    PyObject * moose_setClock(PyObject * dummy, PyObject * args)
    {
        unsigned int tick;
        double dt;
        if(!PyArg_ParseTuple(args, "Id:moose_setClock", &tick, &dt)){
            return NULL;
        }
        if (dt < 0){
            PyErr_SetString(PyExc_ValueError, "dt must be positive.");
            return NULL;
        }
        SHELLPTR->doSetClock(tick, dt);
        Py_RETURN_NONE;
    }

    PyDoc_STRVAR(moose_start_documentation,
                 "start(t) -> None\n"
                 "\n"
                 "Run simulation for `t` time. Advances the simulator clock by `t`\n"
                 "time.\n"                 
                 "\n"
                 "After setting up a simulation, YOU MUST CALL MOOSE.REINIT() before\n"
                 "CALLING MOOSE.START() TO EXECUTE THE SIMULATION. Otherwise, the\n"
                 "simulator behaviour will be undefined. Once moose.reinit() has been\n"
                 "called, you can call moose.start(t) as many time as you like. This\n"
                 "will continue the simulation from the last state for `t` time.\n"
                 "\n"
                 "\nParameters\n"
                 "----------\n"
                 "t : float\n"
                 "\tduration of simulation.\n"
                 "\n"
                 "Returns\n"
                 "--------\n"
                 "\tNone\n"
                 "\n"
                 "See also\n"
                 "--------\n"
                 "moose.reinit : (Re)initialize simulation\n"
                 "\n"
                 );
    PyObject * moose_start(PyObject * dummy, PyObject * args)
    {
        double runtime;
        if(!PyArg_ParseTuple(args, "d:moose_start", &runtime)){
            return NULL;
        }
        if (runtime <= 0.0){
            PyErr_SetString(PyExc_ValueError, "simulation runtime must be positive.");
            return NULL;
        }
        Py_BEGIN_ALLOW_THREADS
                SHELLPTR->doStart(runtime);
        Py_END_ALLOW_THREADS
                Py_RETURN_NONE;
    }

    PyDoc_STRVAR(moose_reinit_documentation,
                 "reinit() -> None\n"
                 "\n"
                 "Reinitialize simulation.\n"
                 "\n"
                 "This function (re)initializes moose simulation. It must be called\n"
                 "before you start the simulation (see moose.start). If you want to\n"
                 "continue simulation after you have called moose.reinit() and\n"
                 "moose.start(), you must NOT call moose.reinit() again. Calling\n"
                 "moose.reinit() again will take the system back to initial setting\n"
                 "(like clear out all data recording tables, set state variables to\n"
                 "their initial values, etc.\n"
                 "\n");
    PyObject * moose_reinit(PyObject * dummy, PyObject * args)
    {
        SHELLPTR->doReinit();
        Py_RETURN_NONE;
    }
    PyObject * moose_stop(PyObject * dummy, PyObject * args)
    {
        SHELLPTR->doStop();
        Py_RETURN_NONE;
    }
    PyObject * moose_isRunning(PyObject * dummy, PyObject * args)
    {
        return Py_BuildValue("i", SHELLPTR->isRunning());
    }

    PyObject * moose_exists(PyObject * dummy, PyObject * args)
    {
        char * path;
        if (!PyArg_ParseTuple(args, "s", &path)){
            return NULL;
        }
        return Py_BuildValue("i", Id(path) != Id() || string(path) == "/" || string(path) == "/root");
    }

    //Harsha : For writing genesis file to sbml
    PyObject * moose_writeSBML(PyObject * dummy, PyObject * args)
    {
        char * fname = NULL, * modelpath = NULL;
        if(!PyArg_ParseTuple(args, "ss:moose_writeSBML", &fname, &modelpath)){
            return NULL;
        }        
        int ret = SHELLPTR->doWriteSBML(string(fname), string(modelpath));
        return Py_BuildValue("i", ret);
    }

    PyObject * moose_readSBML(PyObject * dummy, PyObject * args)
    {
      char * fname = NULL, * modelpath = NULL, * solverclass = NULL;
      if(!PyArg_ParseTuple(args, "ss|s:moose_readSBML", &fname, &modelpath, &solverclass)){
	return NULL;
      }
      //Id ret = SHELLPTR->doReadSBML(string(fname), string(modelpath));
      //return Py_BuildValue("i", ret);
      _Id * model = (_Id*)PyObject_New(_Id, &IdType);
      //model->id_ = SHELLPTR->doReadSBML(string(fname), string(modelpath), string(solverclass));
      if (!solverclass){
	model->id_ = SHELLPTR->doReadSBML(string(fname), string(modelpath));
      } else {
	model->id_ = SHELLPTR->doReadSBML(string(fname), string(modelpath), string(solverclass));
      }
      
      if (model->id_ == Id()){
	Py_XDECREF(model);
	PyErr_SetString(PyExc_IOError, "could not load model");
	return NULL;
      }
      PyObject * ret = reinterpret_cast<PyObject*>(model);
      return ret;
    }

    PyDoc_STRVAR(moose_loadModel_documentation,
                 "loadModel(filename, modelpath, solverclass) -> moose.ematrix\n"
                 "\n"
                 "Load model from a file to a specified path.\n"
                 "\n"
                 "\nParameters\n"
                 "----------\n"
                 "filename : str\n"
                 "\tmodel description file.\n"
                 "modelpath : str\n"
                 "\tmoose path for the top level element of the model to be created.\n"
                 "\tsolverclass : str\n"
                 "\t(optional) solver type to be used for simulating the model.\n"
                 "\n"
                 "Returns\n"
                 "-------\n"
                 "ematrix instance refering to the loaded model container.\n"
                 );
    
    PyObject * moose_loadModel(PyObject * dummy, PyObject * args)
    {
        char * fname = NULL, * modelpath = NULL, * solverclass = NULL;

        if(!PyArg_ParseTuple(args, "ss|s:moose_loadModel", &fname, &modelpath, &solverclass)){
	  cout << "here in moose load";
            return NULL;
        }
        _Id * model = (_Id*)PyObject_New(_Id, &IdType);
        if (!solverclass){
            model->id_ = SHELLPTR->doLoadModel(string(fname), string(modelpath));
        } else {
            model->id_ = SHELLPTR->doLoadModel(string(fname), string(modelpath), string(solverclass));
        }
        if (model->id_ == Id()){
            Py_XDECREF(model);
            PyErr_SetString(PyExc_IOError, "could not load model");
            return NULL;
        }
        PyObject * ret = reinterpret_cast<PyObject*>(model);
        return ret;
    }

    PyDoc_STRVAR(moose_saveModel_documentation,
                 "saveModel(source, fileame)\n"
                 "\n"
                 "Save model rooted at `source` to file `filename`.\n"
                 "\n"
                 "\nParameters\n"
                 "----------\n"
                 "source: ematrix or element or str\n"
                 "\troot of the model tree\n"
                 "\n"
                 "filename: str\n"
                 "\tdestination file to save the model in.\n"
                 "\n"
                 "Returns\n"
                 "-------\n"
                 "None\n"
                 "\n");

    PyObject * moose_saveModel(PyObject * dummy, PyObject * args)
    {
        char * filename = NULL;
        PyObject * source = NULL;
        Id model;
        if (!PyArg_ParseTuple(args, "Os: moose_saveModel", &source, &filename)){
            return NULL;
        }
        if (PyString_Check(source)){
            char * srcPath = PyString_AsString(source);
            if (!srcPath){
                return NULL;
            }
            model = Id(string(srcPath));
        } else if (Id_SubtypeCheck(source)){
            model = ((_Id*)source)->id_;
        } else if (ObjId_SubtypeCheck(source)){
            model = ((_ObjId*)source)->oid_.id;
        } else {
            PyErr_SetString(PyExc_TypeError, "moose_saveModel: need an ematrix, element or string for first argument.");
            return NULL;
        }
        SHELLPTR->doSaveModel(model, filename);
        Py_RETURN_NONE;
    }
    
    PyObject * moose_setCwe(PyObject * dummy, PyObject * args)
    {
        PyObject * element = NULL;
        char * path = NULL;
        Id id;
        if (PyTuple_Size(args) == 0){
            id = Id("/");
        } else if(PyArg_ParseTuple(args, "s:moose_setCwe", &path)){
            id = Id(string(path));
        } else if (PyArg_ParseTuple(args, "O:moose_setCwe", &element)){
            PyErr_Clear();
            if (PyObject_IsInstance(element, (PyObject*)&IdType)){
                id = (reinterpret_cast<_Id*>(element))->id_;
            } else if (PyObject_IsInstance(element, (PyObject*)&ObjIdType)){
                id = (reinterpret_cast<_ObjId*>(element))->oid_.id;                    
            } else {
                PyErr_SetString(PyExc_NameError, "setCwe: Argument must be an ematrix or element");
                return NULL;
            }
        } else {
            return NULL;
        }
        if (!Id::isValid(id)){
            RAISE_INVALID_ID(NULL, "moose_setCwe");
        }
        SHELLPTR->setCwe(id);
        Py_RETURN_NONE;
    }

    PyObject * moose_getCwe(PyObject * dummy, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":moose_getCwe")){
            return NULL;
        }
        _Id * cwe = (_Id*)PyObject_New(_Id, &IdType);
        cwe->id_ = SHELLPTR->getCwe();        
        PyObject * ret = (PyObject*)cwe;
        return ret;
    }

    PyDoc_STRVAR(moose_connect_documentation,
                 "connect(src, src_field, dest, dest_field, message_type) -> bool\n"
                 "\n"
                 "Create a message between `src_field` on `src` object to `dest_field`\n"
                 "on `dest` object.\n"
                 "\n"
                 "Parameters\n"
                 "----------\n"
                 "src : element, ematrix or string\n"
                 "\tthe source object\n"
                 "src_field : str\n"
                 "\tthe source field name. Fields listed under `srcFinfo` and\n"
                 "`sharedFinfo` qualify for this.\n"
                 "dest : element, ematrix or string/\n"
                 "\tthe destination object.\n"
                 "dest_field : str\n"
                 "\tthe destination field name. Fields listed under `destFinfo`\n"
                 "and `sharedFinfo` qualify for this.\n"
                 "message_type : str (optional)\n"
                 "\tType of the message. Can be `Single`, `OneToOne`, `OneToAll`.\n"
                 "If not specified, it defaults to `Single`.\n"
                 "\n"
                 "Returns\n"
                 "-------\n"
                 "element of the message-manager for the newly created message.\n"
                 "\n"
                 "Example\n"
                 "-------\n"
                 "Connect the output of a pulse generator to the input of a spike\n"
                 "generator:\n"
                 "\n"
                 "~~~~\n"
                 ">>> pulsegen = moose.PulseGen('pulsegen')\n"
                 ">>> spikegen = moose.SpikeGen('spikegen')\n"
                 ">>> moose.connect(pulsegen, 'outputOut', spikegen, 'Vm')\n"
                 "1\n"
                 "~~~~\n"
                 "\n"

                 );
    
    PyObject * moose_connect(PyObject * dummy, PyObject * args)
    {
        PyObject * srcPtr = NULL, * destPtr = NULL;
        char * srcField = NULL, * destField = NULL, * msgType = NULL;
        static char default_msg_type[] = "Single";
        if(!PyArg_ParseTuple(args, "OsOs|s:moose_connect", &srcPtr, &srcField, &destPtr, &destField, &msgType)){
            return NULL;
        }
        if (msgType == NULL){
            msgType = default_msg_type;
        }
        ObjId dest, src;
        if (ObjId_SubtypeCheck(srcPtr)){
            _ObjId * _src = reinterpret_cast<_ObjId*>(srcPtr);
            src = _src->oid_;            
        } else if (Id_SubtypeCheck(srcPtr)){
            _Id * _src = reinterpret_cast<_Id*>(srcPtr);
            src = ObjId(_src->id_);
        } else if (PyString_Check(srcPtr)){
            char * _src = PyString_AsString(srcPtr);
            src = ObjId(string(_src));
        } else {
            PyErr_SetString(PyExc_TypeError, "source does not resolve to an element.");
            return NULL;
        }
        if (ObjId_SubtypeCheck(destPtr)){
            _ObjId * _dest = reinterpret_cast<_ObjId*>(destPtr);
            dest = _dest->oid_;            
        } else if (Id_SubtypeCheck(destPtr)){
            _Id * _dest = reinterpret_cast<_Id*>(destPtr);
            dest = ObjId(_dest->id_);
        } else if (PyString_Check(destPtr)){
            char * _dest = PyString_AsString(destPtr);
            dest = ObjId(string(_dest));
        } else {
            PyErr_SetString(PyExc_TypeError, "target does not resolve to an element.");
            return NULL;
        }
        if (!Id::isValid(dest.id) || !Id::isValid(src.id)){
            RAISE_INVALID_ID(NULL, "moose_connect");
        }
        ObjId mid = SHELLPTR->doAddMsg(msgType, src, string(srcField), dest, string(destField));
        if ( mid == ObjId() ){
            PyErr_SetString(PyExc_NameError, "check field names and type compatibility.");
            return NULL;
        }
        _ObjId * msgMgrId = (_ObjId*)PyObject_New(_ObjId, &ObjIdType);
        msgMgrId->oid_ = mid;
        return (PyObject*) msgMgrId;
    }

    
    PyDoc_STRVAR(moose_getFieldDict_documentation,
                 "getFieldDict(className, finfoType) -> dict\n"
                 "\n"
                 "Get dictionary of field names and types for specified class.\n"
                 "Parameters\n"
                 "-----------\n"
                 "className : str\n"
                 "\tMOOSE class to find the fields of.\n"
                 "finfoType : str (optional)\n"
                 "\tFinfo type of the fields to find. If empty or not specified, all\n"
                 "fields will be retrieved.\n"
                 "note: This behaviour is different from `getFieldNames` where only\n"
                 "`valueFinfo`s are returned when `finfoType` remains unspecified.\n"
                 "\n"
                 "Example\n"
                 "-------\n"
                 "List all the source fields on class Neutral:\n"
                 "~~~~\n"
                 ">>> moose.getFieldDict('Neutral', 'srcFinfo')\n"
                 "{'childMsg': 'int'}\n"
                 "~~~~\n"
                 "\n");
    PyObject * moose_getFieldDict(PyObject * dummy, PyObject * args)
    {
        char * className = NULL;
        char * fieldType = NULL;
        if (!PyArg_ParseTuple(args, "s|s:moose_getFieldDict", &className, &fieldType)){
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
            if (getFieldDict(classId, string(fieldType), fields, types) == 0){
                PyErr_SetString(PyExc_ValueError, "Invalid finfo type.");
                return NULL;
            }
        } else {
            for (const char ** ptr = finfoTypes; *ptr != NULL; ++ptr){
                if (getFieldDict(classId, string(*ptr), fields, types) == 0){
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
            PyObject * value = Py_BuildValue("s", types[ii].c_str());
            if (value == NULL || PyDict_SetItemString(ret, fields[ii].c_str(), value) == -1){
                Py_XDECREF(ret);
                Py_XDECREF(value);
                return NULL;
            }
        }
        return ret;
    }

    PyObject * moose_getField(PyObject * dummy, PyObject * args)
    {
        PyObject * pyobj;
        const char * field;
        const char * type;
        if (!PyArg_ParseTuple(args, "Oss:moose_getfield", &pyobj, &field, &type)){
            return NULL;
        }
        if (!PyObject_IsInstance(pyobj, (PyObject*)&ObjIdType)){
            PyErr_SetString(PyExc_TypeError, "moose.getField(element, fieldname, fieldtype): First argument must be an instance of element or its subclass");
            return NULL;
        }
        string fname(field), ftype(type);
        ObjId oid = ((_ObjId*)pyobj)->oid_;
        if (!Id::isValid(oid.id)){
            RAISE_INVALID_ID(NULL, "moose_getField");
        }
        // Let us do this version using brute force. Might be simpler than getattro.
        if (ftype == "char"){
            char value =Field<char>::get(oid, fname);
            return PyInt_FromLong(value);            
        } else if (ftype == "double"){
            double value = Field<double>::get(oid, fname);
            return PyFloat_FromDouble(value);
        } else if (ftype == "float"){
            float value = Field<float>::get(oid, fname);
            return PyFloat_FromDouble(value);
        } else if (ftype == "int"){
            int value = Field<int>::get(oid, fname);
            return PyInt_FromLong(value);
        } else if (ftype == "string"){
            string value = Field<string>::get(oid, fname);
            return PyString_FromString(value.c_str());
        } else if (ftype == "unsigned int" || ftype == "unsigned" || ftype == "uint"){
            unsigned int value = Field<unsigned int>::get(oid, fname);
            return PyInt_FromLong(value);
        } else if (ftype == "Id"){
            _Id * value = (_Id*)PyObject_New(_Id, &IdType);
            value->id_ = Field<Id>::get(oid, fname);
            return (PyObject*) value;
        } else if (ftype == "ObjId"){
            _ObjId * value = (_ObjId*)PyObject_New(_ObjId, &ObjIdType);
            value->oid_ = Field<ObjId>::get(oid, fname);
            return (PyObject*)value;
        } else if (ftype == "vector<int>"){
            vector<int> value = Field< vector < int > >::get(oid, fname);
            PyObject * ret = PyTuple_New((Py_ssize_t)value.size());
                
            for (unsigned int ii = 0; ii < value.size(); ++ ii ){     
                PyObject * entry = Py_BuildValue("i", value[ii]); 
                if (!entry || PyTuple_SetItem(ret, (Py_ssize_t)ii, entry)){ 
                    Py_XDECREF(ret);
                    ret = NULL;                                 
                    break;                                      
                }                                               
            }
            return ret;
        } else if (ftype == "vector<double>"){
            vector<double> value = Field< vector < double > >::get(oid, fname);
            PyObject * ret = PyTuple_New((Py_ssize_t)value.size());
                
            for (unsigned int ii = 0; ii < value.size(); ++ ii ){     
                PyObject * entry = Py_BuildValue("f", value[ii]); 
                if (!entry || PyTuple_SetItem(ret, (Py_ssize_t)ii, entry)){ 
                    Py_XDECREF(ret);                                  
                    ret = NULL;                                 
                    break;                                      
                }                                               
            }
            return ret;
        } else if (ftype == "vector<float>"){
            vector<float> value = Field< vector < float > >::get(oid, fname);
            PyObject * ret = PyTuple_New((Py_ssize_t)value.size());
                
            for (unsigned int ii = 0; ii < value.size(); ++ ii ){     
                PyObject * entry = Py_BuildValue("f", value[ii]); 
                if (!entry || PyTuple_SetItem(ret, (Py_ssize_t)ii, entry)){ 
                    Py_XDECREF(ret);                                  
                    ret = NULL;                                 
                    break;                                      
                }                                            
            }
            return ret;
        } else if (ftype == "vector<string>"){
            vector<string> value = Field< vector < string > >::get(oid, fname);
            PyObject * ret = PyTuple_New((Py_ssize_t)value.size());
                
            for (unsigned int ii = 0; ii < value.size(); ++ ii ){     
                PyObject * entry = Py_BuildValue("s", value[ii].c_str()); 
                if (!entry || PyTuple_SetItem(ret, (Py_ssize_t)ii, entry)){ 
                    Py_XDECREF(ret);                                  
                    return NULL;                                 
                }                                            
            }
            return ret;
        } else if (ftype == "vector<Id>"){
            vector<Id> value = Field< vector < Id > >::get(oid, fname);
            PyObject * ret = PyTuple_New((Py_ssize_t)value.size());
                
            for (unsigned int ii = 0; ii < value.size(); ++ ii ){
                _Id * entry = PyObject_New(_Id, &IdType);
                entry->id_ = value[ii]; 
                if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){ 
                    Py_XDECREF(ret);                                  
                    return NULL;                                 
                }                                            
            }
            return ret;
        } else if (ftype == "vector<ObjId>"){
            vector<ObjId> value = Field< vector < ObjId > >::get(oid, fname);
            PyObject * ret = PyTuple_New((Py_ssize_t)value.size());
                
            for (unsigned int ii = 0; ii < value.size(); ++ ii ){
                _ObjId * entry = PyObject_New(_ObjId, &ObjIdType);
                entry->oid_ = value[ii]; 
                if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){ 
                    Py_XDECREF(ret);                                  
                    return NULL;                                 
                }                                            
            }
            return ret;
        }
        PyErr_SetString(PyExc_TypeError, "Field type not handled.");
        return NULL;
    }
        
    PyDoc_STRVAR(moose_seed_documentation, 
                 "moose.seed(seedvalue) -> None\n"
                 "\n"
                 "Reseed MOOSE random number generator.\n"
                 "\n"
                 "\nParameters\n"
                 "----------\n"
                 "seed: int\n"
                 "\tOptional value to use for seeding. If 0, a random seed is"
                 "\n\tautomatically created using the current system time and other"
                 "\n\tinformation. If not specified, it defaults to 0."
                 "\n");
    
    PyObject * moose_seed(PyObject * dummy, PyObject * args)
    {
        long seed = 0;
        if (!PyArg_ParseTuple(args, "|l", &seed)){
            return NULL;
        }
        mtseed(seed);
        Py_RETURN_NONE;
    }

    PyDoc_STRVAR(moose_wildcardFind_documentation,
                 "moose.wildcardFind(expression) -> tuple of ematrices.\n"
                 "\n"
                 "Find an object by wildcard.\n"
                 "\n"
                 "\nParameters\n"
                 "----------\n"
                 "expression: str\n"
                 "\tMOOSE allows wildcard expressions of the form\n"
                 "\t{PATH}/{WILDCARD}[{CONDITION}]\n"
                 "\twhere {PATH} is valid path in the element tree.\n"
                 "\t{WILDCARD} can be `#` or `##`.\n"
                 "\t`#` causes the search to be restricted to the children of the\n"
                 "\telement specified by {PATH}.\n"
                 "\t`##` makes the search to recursively go through all the descendants\n"
                 "\tof the {PATH} element.\n"
                 "\t{CONDITION} can be\n"
                 "\tTYPE={CLASSNAME} : an element satisfies this condition if it is of\n"
                 "\tclass {CLASSNAME}.\n"
                 "\tISA={CLASSNAME} : alias for TYPE={CLASSNAME}\n"
                 "\tCLASS={CLASSNAME} : alias for TYPE={CLASSNAME}\n"
                 "\tFIELD({FIELDNAME}){OPERATOR}{VALUE} : compare field {FIELDNAME} with\n"
                 "\t{VALUE} by {OPERATOR} where {OPERATOR} is a comparison operator (=,\n"
                 "\t!=, >, <, >=, <=).\n"
                 "\tFor example, /mymodel/##[FIELD(Vm)>=-65] will return a list of all\n"
                 "\tthe objects under /mymodel whose Vm field is >= -65.\n"
                 "\n");
    PyObject * moose_wildcardFind(PyObject * dummy, PyObject * args)
    {
        vector <Id> objects;
        char * wildcard_path = NULL;
        if (!PyArg_ParseTuple(args, "s:moose.wildcardFind", &wildcard_path)){
            return NULL;
        }
        SHELLPTR->wildcard(string(wildcard_path), objects);
        PyObject * ret = PyTuple_New(objects.size());
        if (ret == NULL){
            PyErr_SetString(PyExc_RuntimeError, "moose.wildcardFind: failed to allocate new tuple.");
            return NULL;
        }
            
        for (unsigned int ii = 0; ii < objects.size(); ++ii){
            _Id * entry = PyObject_New(_Id, &IdType);                       
            if (!entry){
                Py_XDECREF(ret);
                PyErr_SetString(PyExc_RuntimeError, "moose.wildcardFind: failed to allocate new ematrix.");
                return NULL;
            }
            entry->id_ = objects[ii];
            if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){
                Py_XDECREF(entry);
                Py_XDECREF(ret);
                return NULL;
            }
        }
        return ret;
    }
    // This should not be required or accessible to the user. Put here
    // for debugging threading issue.
    PyObject * moose_quit(PyObject * dummy)
    {
        finalize();
        cout << "Quitting MOOSE." << endl;
        Py_RETURN_NONE;
    }

    /// Go through all elements under /classes and ask for defining a
    /// Python class for it.
    int defineAllClasses(PyObject * module_dict)
    {
        static vector <Id> classes(Field< vector<Id> >::get(ObjId("/classes"),
                                                            "children"));
        for (unsigned ii = 0; ii < classes.size(); ++ii){
            const string& class_name = classes[ii].element()->getName();
            const Cinfo * cinfo = Cinfo::find(class_name);
            if (!cinfo){
                cerr << "Error: no cinfo found with name " << class_name << endl;
                return 0;
            }
            if (!define_class(module_dict, cinfo)){
                return 0;
            }
        }
        return 1;
    }

    // An attempt to define classes dynamically
    // http://stackoverflow.com/questions/8066438/how-to-dynamically-create-a-derived-type-in-the-python-c-api
    // gives a clue We pass class_name in stead of class_id because we
    // have to recursively call this function using the base class
    // string.
    PyDoc_STRVAR(moose_Class_documentation,
                 "*-----------------------------------------------------------------*\n"
                 "* This is Python generated documentation.                         *\n"
                 "* Use moose.doc('classname') to display builtin documentation for *\n"
                 "* class `classname`.                                              *\n"
                 "* Use moose.doc('classname.fieldname') to display builtin         *\n"
                 "* documentationfor `field` in class `classname`.                  *\n"
                 "*-----------------------------------------------------------------*\n"
                 );

    int define_class(PyObject * module_dict, const Cinfo * cinfo)
    {
        const string& class_name = cinfo->name();
        map <string, PyTypeObject * >::iterator existing =
                get_moose_classes().find(class_name);
        if (existing != get_moose_classes().end()){
            return 1;
        }
        const Cinfo* base = cinfo->baseCinfo();
        if (base && !define_class(module_dict, base)){
            return 0;
        }
        PyTypeObject * new_class =
                (PyTypeObject*)PyType_Type.tp_alloc(&PyType_Type, 0);
        // Py_TYPE(new_class) = &PyType_Type;
        new_class->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        // should we avoid Py_TPFLAGS_HEAPTYPE as it imposes certain
        // limitations:
        // http://mail.python.org/pipermail/python-dev/2009-July/090921.html
        // But otherwise somehow GC tries tp_traverse on these classes
        // (even when I unset Py_TPFLAGS_HAVE_GC) and fails the
        // assertion in debug build of Python:
        //
        // python: Objects/typeobject.c:2683: type_traverse: Assertion `type->tp_flags & Py_TPFLAGS_HEAPTYPE' failed.
        //
        // In released versions of Python there is a crash at
        // Py_Finalize().
        //
        // Also if HEAPTYPE is true, then help(classname) causes a
        // segmentation fault as it tries to convert the class object
        // to a heaptype object (resulting in an invalid pointer). If
        // heaptype is not set it uses tp_name to print the help.
        // Py_SIZE(new_class) = sizeof(_ObjId);        
        string str = "moose." + class_name;
        new_class->tp_name = (char *)calloc(str.length()+1,
                                            sizeof(char));
        strncpy(const_cast<char*>(new_class->tp_name), str.c_str(),
                str.length());
        new_class->tp_doc = moose_Class_documentation;
        // strncpy(new_class->tp_doc, moose_Class_documentation, strlen(moose_Class_documentation));
        map<string, PyTypeObject *>::iterator base_iter =
                get_moose_classes().find(cinfo->getBaseClass());
        if (base_iter == get_moose_classes().end()){
            new_class->tp_base = &ObjIdType;
        } else {
            new_class->tp_base = base_iter->second;
        }
        Py_INCREF(new_class->tp_base);
        // Define all the lookupFields
        if (!define_lookupFinfos(cinfo)){            
            return 0;
        }
        // Define the destFields
        if (!define_destFinfos(cinfo)){
            return 0;
        }

        // Define the element fields
        if (!define_elementFinfos(cinfo)){
            return 0;
        }
        // #ifndef NDEBUG
        //         cout << "Get set defs:" << class_name << endl;
        //         for (unsigned int ii = 0; ii < get_getsetdefs()[class_name].size(); ++ii){
        //             cout << ii;
        //             if (get_getsetdefs()[class_name][ii].name != NULL){
        //                 cout << ": " << get_getsetdefs()[class_name][ii].name;
        //             } else {
        //                 cout << "Empty";
        //             }
        //             cout << endl;
        //         }
        //         cout << "End getsetdefs: " << class_name << endl;
        // #endif
        // The getsetdef array must be terminated with empty objects.
        PyGetSetDef empty;
        empty.name = NULL;
        get_getsetdefs()[class_name].push_back(empty);
        new_class->tp_getset = &(get_getsetdefs()[class_name][0]);
        // Cannot do this for HEAPTYPE ?? but pygobject.c does this in
        // pygobject_register_class
        if (PyType_Ready(new_class) < 0){
            cerr << "Fatal error: Could not initialize class '" << class_name
                 << "'" << endl;
            return 0;
        }
        get_moose_classes().insert(pair<string, PyTypeObject*> (class_name, new_class));
        Py_INCREF(new_class);
        // PyDict_SetItemString(new_class->tp_dict, "__module__", PyString_FromString("moose"));
        // string doc = const_cast<Cinfo*>(cinfo)->getDocs();
        // PyDict_SetItemString(new_class->tp_dict, "__doc__", PyString_FromString(" \0"));
        // PyDict_SetItemString(module_dict, class_name.c_str(), (PyObject *)new_class);
        return 1;                
    }
    
    PyObject * moose_ObjId_get_destField_attr(PyObject * self, void * closure)
    {
        if (!PyObject_IsInstance(self, (PyObject*)&ObjIdType)){
            PyErr_SetString(PyExc_TypeError, "First argument must be an instance of element");
            return NULL;
        }
        _ObjId * obj = (_ObjId*)self;
        if (!Id::isValid(obj->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_get_destField_attr");
        }
        char * name = NULL;
        if (!PyArg_ParseTuple((PyObject *)closure,
                              "s:_get_destField: "
                              "expected a string in getter closure.",
                              &name)){
            return NULL;
        }
        // If the DestField already exists, return it
        string full_name = obj->oid_.path() +
                "." + string(name);
        map<string, PyObject * >::iterator it = get_inited_destfields().find(full_name);
        if (it != get_inited_destfields().end()){
            Py_XINCREF(it->second);
            return it->second;
        }
        PyObject * args = PyTuple_New(2);
                
        PyTuple_SetItem(args, 0, self);
        Py_INCREF(self); // compensate for reference stolen by PyTuple_SetItem
        PyTuple_SetItem(args, 1, PyString_FromString(name));
        _Field * ret = PyObject_New(_Field, &moose_DestField);
        if (moose_DestField.tp_init((PyObject*)ret, args, NULL) == 0){
            Py_XDECREF(args);
            // I thought PyObject_New creates a new ref, but without
            // the following XINCREF, the destinfo gets gc-ed.
            Py_XINCREF(ret);
            get_inited_destfields()[full_name] =  (PyObject*)ret;
            return (PyObject*)ret;
        }
        Py_XDECREF((PyObject*)ret);
        Py_XDECREF(args);
        return NULL;
    }
    
    
    int define_destFinfos(const Cinfo * cinfo)
    {
        static char * doc = "Destination field";
        const string& class_name = cinfo->name();
        // Create methods for destFinfos. The tp_dict is initialized by
        // PyType_Ready. So we insert the dynamically generated
        // methods after that.        
        vector <PyGetSetDef>& vec = get_getsetdefs()[class_name];

        // We do not know the final number of user-accessible
        // destFinfos as we have to ignore the destFinfos starting
        // with get/set. So use a vector instead of C array.
        size_t curr_index = vec.size();
        for (unsigned int ii = 0; ii < cinfo->getNumDestFinfo(); ++ii){
            Finfo * destFinfo = const_cast<Cinfo*>(cinfo)->getDestFinfo(ii);
            const string& destFinfo_name = destFinfo->name();
            // get_{xyz} and set_{xyz} are internal destFinfos for
            // accessing valueFinfos. Ignore them.
            if (destFinfo_name.find("get_") == 0 || destFinfo_name.find("set_") == 0){
                continue;
            }
            PyGetSetDef destFieldGetSet;
            vec.push_back(destFieldGetSet);
            vec[curr_index].name = (char*)calloc(destFinfo_name.size() + 1, sizeof(char));
            strncpy(vec[curr_index].name,
                    const_cast<char*>(destFinfo_name.c_str()),
                    destFinfo_name.size());
            vec[curr_index].doc = doc;
            vec[curr_index].get = (getter)moose_ObjId_get_destField_attr;
            PyObject * args = PyTuple_New(1);
            
            if (args == NULL){
                cerr << "moosemodule.cpp: define_destFinfos: Failed to allocate tuple" << endl;
                return 0;
            }
            PyTuple_SetItem(args, 0, PyString_FromString(destFinfo_name.c_str()));
            vec[curr_index].closure = (void*)args;
            ++curr_index;
        } // ! for
        
        return 1;
    }

    /**
       Try to obtain a LookupField object for a specified
       lookupFinfo. The first item in `closure` must be the name of
       the LookupFinfo - {fieldname}. The LookupField is identified by
       {path}.{fieldname} where {path} is the unique path of the moose
       element `self`. We look for an already initialized LookupField
       object for this identifier and return if one is
       found. Otherwise, we create a new LookupField object and buffer
       it in a map before returning.
       
     */
    PyObject * moose_ObjId_get_lookupField_attr(PyObject * self,
                                                       void * closure)
    {
        if (!PyObject_IsInstance(self, (PyObject*)&ObjIdType)){
            PyErr_SetString(PyExc_TypeError,
                            "First argument must be an instance of element");
            return NULL;
        }
        _ObjId * obj = (_ObjId*)self;
        if (!Id::isValid(obj->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_get_lookupField_attr");
        }
        char * name = NULL;
        if (!PyArg_ParseTuple((PyObject *)closure,
                              "s:moose_ObjId_get_lookupField_attr: expected a string in getter closure.",
                              &name)){
            return NULL;
        }
        assert(name);
        // If the LookupField already exists, return it
        string full_name = obj->oid_.path() + "." + string(name);
        map<string, PyObject * >::iterator it = get_inited_lookupfields().find(full_name);
        if (it != get_inited_lookupfields().end()){
            Py_XINCREF(it->second);
            return it->second;
        }
        
        // Create a new instance of LookupField `name` and set it as
        // an attribute of the object self.

        // Create the argument for init method of LookupField.  Thisx
        // will be (fieldname, self)
        PyObject * args = PyTuple_New(2);
                
        PyTuple_SetItem(args, 0, self);
        Py_XINCREF(self); // compensate for stolen ref
        PyTuple_SetItem(args, 1, PyString_FromString(name));
        _Field * ret = PyObject_New(_Field, &moose_LookupField);
        if (moose_LookupField.tp_init((PyObject*)ret, args, NULL) == 0){
            Py_XDECREF(args);
            get_inited_lookupfields()[full_name] =  (PyObject*)ret;
            // I thought PyObject_New creates a new ref, but without
            // the following XINCREF, the lookupfinfo gets gc-ed.
            Py_XINCREF(ret);
            return (PyObject*)ret;
        }
        Py_XDECREF((PyObject*)ret);
        Py_XDECREF(args);
        return NULL;
    }

    PyObject * oid_to_element(ObjId oid)
    {
        string classname = Field<string>::get(oid, "className");
        map<string, PyTypeObject *>::iterator it = get_moose_classes().find(classname);
        if (it == get_moose_classes().end()){
            return NULL;
        }
        PyTypeObject * pyclass = it->second;
        _ObjId * new_obj = PyObject_New(_ObjId, pyclass);
        new_obj->oid_ = oid;
        Py_XINCREF(new_obj);
        return (PyObject*)new_obj;
    }

    PyDoc_STRVAR(moose_element_documentation,
                 "moose.element(arg) -> moose object\n"
                 "\n"
                 "Convert a path or an object to the appropriate builtin moose class\n"
                 "instance\n"
                 "Parameters\n"
                 "----------\n"
                 "arg: str or ematrix or moose object\n"
                 "path of the moose element to be converted or another element (possibly\n"
                 "available as a superclass instance).\n"
                 "\n"
                 "Returns\n"
                 "-------\n"
                 "An element of the moose builtin class the specified object belongs\n"
                 "to.\n"
                 "\n");
    PyObject * moose_element(PyObject* dummy, PyObject * args)
    {
        char * path = NULL;
        PyObject * obj = NULL;
        ObjId oid;
        if (PyArg_ParseTuple(args, "s", &path)){
            oid = ObjId(path);
            //            cout << "Original Path " << path << ", Element Path: " << oid.path() << endl;
            if (ObjId::bad() == oid){
                PyErr_SetString(PyExc_ValueError, "moose_element: path does not exist");
                return NULL;
            }
            PyObject * new_obj = oid_to_element(oid);
            if (new_obj){
                return new_obj;
            }
            PyErr_SetString(PyExc_TypeError, "moose_element: unknown class");
            return NULL;
        }
        PyErr_Clear();
        if (!PyArg_ParseTuple(args, "O", &obj)){
            PyErr_SetString(PyExc_TypeError, "moose_element: argument must be a path or an existing element or an ematrix");
            return NULL;
        }
        if (PyObject_IsInstance(obj, (PyObject*)&ObjIdType)){
            oid = ((_ObjId*)obj)->oid_;
        } else if (PyObject_IsInstance(obj, (PyObject*)&IdType)){
            oid = ObjId(((_Id*)obj)->id_);
        } else if (ElementField_SubtypeCheck(obj)){
            oid = ObjId(((_Id*)moose_ElementField_getId((_Field*)obj, NULL))->id_);
        }
        if (oid == ObjId::bad()){
            PyErr_SetString(PyExc_TypeError, "moose_element: cannot convert to moose element.");
            return NULL;
        }
        PyObject * new_obj = oid_to_element(oid);
        if (!new_obj){
            PyErr_SetString(PyExc_RuntimeError, "moose_element: not a moose class.");
        }
        return new_obj;
    }
    
    int define_lookupFinfos(const Cinfo * cinfo)
    {
        static char * doc = "Lookup field";
        const string & class_name = cinfo->name();
        unsigned int num_lookupFinfos = cinfo->getNumLookupFinfo();
        unsigned int curr_index = get_getsetdefs()[class_name].size();
        for (unsigned int ii = 0; ii < num_lookupFinfos; ++ii){
            const string& lookupFinfo_name = const_cast<Cinfo*>(cinfo)->getLookupFinfo(ii)->name();
            PyGetSetDef getset;
            get_getsetdefs()[class_name].push_back(getset);
            get_getsetdefs()[class_name][curr_index].name = (char*)calloc(lookupFinfo_name.size() + 1, sizeof(char));
            strncpy(get_getsetdefs()[class_name][curr_index].name, const_cast<char*>(lookupFinfo_name.c_str()), lookupFinfo_name.size());
            get_getsetdefs()[class_name][curr_index].doc = doc; //moose_LookupField_documentation;
            get_getsetdefs()[class_name][curr_index].get = (getter)moose_ObjId_get_lookupField_attr;
            PyObject * args = PyTuple_New(1);
            PyTuple_SetItem(args, 0, PyString_FromString(lookupFinfo_name.c_str()));
            get_getsetdefs()[class_name][curr_index].closure = (void*)args;
            ++curr_index;
        }
        return 1;
    }

    PyObject * moose_ObjId_get_elementField_attr(PyObject * self,
                                                        void * closure)
    {
        // if (!PyObject_IsInstance(self, (PyObject*)&ObjIdType)){
        //       PyErr_SetString(PyExc_TypeError,
        //                       "First argument must be an instance of element");
        //       return NULL;
        //   }
        _ObjId * obj = (_ObjId*)self;
        if (!Id::isValid(obj->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_get_elementField_attr");
        }
        char * name = NULL;
        if (!PyArg_ParseTuple((PyObject *)closure,
                              "s:moose_ObjId_get_elementField_attr: expected a string in getter closure.",
                              &name)){
            return NULL;
        }
        // If the ElementField already exists, return it
        string full_name = obj->oid_.path() + "." + string(name);
        // cout << "ElementField fullname: " << full_name << endl;
        map<string, PyObject * >::iterator it = get_inited_elementfields().find(full_name);
        if (it != get_inited_elementfields().end()){
            Py_XINCREF(it->second);
            return it->second;
        }
        
        // Create a new instance of ElementField `name` and set it as
        // an attribute of the object `self`.
        // 1. Create the argument for init method of ElementField.  This
        //   will be (fieldname, self)
        PyObject * args = PyTuple_New(2);                
        PyTuple_SetItem(args, 0, self);
        Py_XINCREF(self); // compensate for stolen ref
        PyTuple_SetItem(args, 1, PyString_FromString(name));
        _Field * ret = PyObject_New(_Field, &moose_ElementField);
        // 2. Now use this arg to actually create the element field.
        if (moose_ElementField.tp_init((PyObject*)ret, args, NULL) == 0){
            Py_XDECREF(args);
            get_inited_elementfields()[full_name] =  (PyObject*)ret;
            // I thought PyObject_New creates a new ref, but without
            // the following XINCREF, the finfo gets gc-ed.
            Py_XINCREF(ret);
            return (PyObject*)ret;
        }
        Py_XDECREF((PyObject*)ret);
        Py_XDECREF(args);
        return NULL;
    }

    int define_elementFinfos(const Cinfo * cinfo)
    {
        static char * doc = "Element field\0";
        const string & class_name = cinfo->name();
        unsigned int num_fieldElementFinfo = cinfo->getNumFieldElementFinfo();
        unsigned int curr_index = get_getsetdefs()[class_name].size();
        for (unsigned int ii = 0; ii < num_fieldElementFinfo; ++ii){
            const string& finfo_name = const_cast<Cinfo*>(cinfo)->getFieldElementFinfo(ii)->name();
            PyGetSetDef getset;
            get_getsetdefs()[class_name].push_back(getset);
            get_getsetdefs()[class_name][curr_index].name = (char*)calloc(finfo_name.size() + 1, sizeof(char));
            strncpy(get_getsetdefs()[class_name][curr_index].name, const_cast<char*>(finfo_name.c_str()), finfo_name.size());
            get_getsetdefs()[class_name][curr_index].doc = doc;
            get_getsetdefs()[class_name][curr_index].get = (getter)moose_ObjId_get_elementField_attr;
            PyObject * args = PyTuple_New(1);
            PyTuple_SetItem(args, 0, PyString_FromString(finfo_name.c_str()));
            get_getsetdefs()[class_name][curr_index].closure = (void*)args;
            ++curr_index;
        }
        return 1;
    }



    /////////////////////////////////////////////////////////////////////
    // Method definitions for MOOSE module
    /////////////////////////////////////////////////////////////////////    
    static PyMethodDef MooseMethods[] = {
        {"element", (PyCFunction)moose_element, METH_VARARGS, moose_element_documentation},
        {"getFieldNames", (PyCFunction)moose_getFieldNames, METH_VARARGS, moose_getFieldNames_documentation},
        {"copy", (PyCFunction)moose_copy, METH_VARARGS|METH_KEYWORDS, moose_copy_documentation},
        {"move", (PyCFunction)moose_move, METH_VARARGS, "Move a ematrix object to a destination."},
        {"delete", (PyCFunction)moose_delete, METH_VARARGS, moose_delete_documentation},
        {"useClock", (PyCFunction)moose_useClock, METH_VARARGS, "Schedule objects on a specified clock"},
        {"setClock", (PyCFunction)moose_setClock, METH_VARARGS, "Set the dt of a clock."},
        {"start", (PyCFunction)moose_start, METH_VARARGS, moose_start_documentation},
        {"reinit", (PyCFunction)moose_reinit, METH_VARARGS, moose_reinit_documentation},
        {"stop", (PyCFunction)moose_stop, METH_VARARGS, "Stop simulation"},
        {"isRunning", (PyCFunction)moose_isRunning, METH_VARARGS, "True if the simulation is currently running."},
        {"exists", (PyCFunction)moose_exists, METH_VARARGS, "True if there is an object with specified path."},
        {"writeSBML", (PyCFunction)moose_writeSBML, METH_VARARGS, "Export biochemical model to an SBML file."},
	{"readSBML",  (PyCFunction)moose_readSBML,  METH_VARARGS, "Import SBML model to Moose."},
        {"loadModel", (PyCFunction)moose_loadModel, METH_VARARGS, moose_loadModel_documentation},
        {"saveModel", (PyCFunction)moose_saveModel, METH_VARARGS, moose_saveModel_documentation},
        {"connect", (PyCFunction)moose_connect, METH_VARARGS, moose_connect_documentation},        
        {"getCwe", (PyCFunction)moose_getCwe, METH_VARARGS, "Get the current working element. 'pwe' is an alias of this function."},
        // {"pwe", (PyCFunction)moose_getCwe, METH_VARARGS, "Get the current working element. 'getCwe' is an alias of this function."},
        {"setCwe", (PyCFunction)moose_setCwe, METH_VARARGS, "Set the current working element. 'ce' is an alias of this function"},
        // {"ce", (PyCFunction)moose_setCwe, METH_VARARGS, "Set the current working element. setCwe is an alias of this function."},
        {"getFieldDict", (PyCFunction)moose_getFieldDict, METH_VARARGS, moose_getFieldDict_documentation},
        {"getField", (PyCFunction)moose_getField, METH_VARARGS,
         "getField(element, field, fieldtype) -- Get specified field of specified type from object ematrix."},
        {"seed", (PyCFunction)moose_seed, METH_VARARGS, moose_seed_documentation},
        {"wildcardFind", (PyCFunction)moose_wildcardFind, METH_VARARGS, moose_wildcardFind_documentation},
        {"quit", (PyCFunction)moose_quit, METH_NOARGS, "Finalize MOOSE threads and quit MOOSE. This is made available for"
         " debugging purpose only. It will automatically get called when moose"
         " module is unloaded. End user should not use this function."},

        {NULL, NULL, 0, NULL}        /* Sentinel */
    };


    
    ///////////////////////////////////////////////////////////
    // module initialization 
    ///////////////////////////////////////////////////////////
    PyDoc_STRVAR(moose_module_documentation,
                 "MOOSE = Multiscale Object-Oriented Simulation Environment.\n"
                 "\n"
                 "Moose is the core of a modern software platform for the simulation\n"
                 "of neural systems ranging from subcellular components and\n"
                 "biochemical reactions to complex models of single neurons, large\n"
                 "networks, and systems-level processes.");

#ifdef PY3K

    int moose_traverse(PyObject *m, visitproc visit, void *arg) {
        Py_VISIT(GETSTATE(m)->error);
        return 0;
    }

    int moose_clear(PyObject *m) {
        Py_CLEAR(GETSTATE(m)->error);
        // I did get a segmentation fault at exit (without running a reinit() or start()) after creating a compartment. After putting the finalize here it went away. But did not reoccur even after commenting it out. Will need closer debugging.
        // - Subha 2012-08-18, 00:36    
        finalize();
        return 0;
    }


    static struct PyModuleDef MooseModuleDef = {
        PyModuleDef_HEAD_INIT,
        "moose", /* m_name */
        moose_module_documentation, /* m_doc */
        sizeof(struct module_state), /* m_size */
        MooseMethods, /* m_methods */
        0, /* m_reload */
        moose_traverse, /* m_traverse */
        moose_clear, /* m_clear */
        NULL /* m_free */
    };

#define INITERROR return NULL
#define MODINIT(name) PyInit_##name()
#else // Python 2
#define INITERROR return
#define MODINIT(name) init##name()
#endif
                 
    PyMODINIT_FUNC MODINIT(_moose)
    {
        clock_t modinit_start = clock();
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();
        // First of all create the Shell.  We convert the environment
        // variables into c-like argv array
        vector<string> args = setup_runtime_env();
        int argc = args.size();
        char ** argv = (char**)calloc(args.size(), sizeof(char*));
        for (int ii = 0; ii < argc; ++ii){
            argv[ii] = (char*)(calloc(args[ii].length()+1, sizeof(char)));
            strncpy(argv[ii], args[ii].c_str(), args[ii].length()+1);            
        }
        PyEval_InitThreads();
        Id shellId = get_shell(argc, argv);
        for (int ii = 1; ii < argc; ++ii){
            free(argv[ii]);
        }
        // Now initialize the module
#ifdef PY3K
	PyObject * moose_module = PyModule_Create(&MooseModuleDef);
#else
        PyObject *moose_module = Py_InitModule3("_moose",
                                                MooseMethods,
                                                moose_module_documentation);
#endif
        if (moose_module == NULL){
            INITERROR;
        }
	struct module_state * st = GETSTATE(moose_module);
        char error[] = "moose.Error";
	st->error = PyErr_NewException(error, NULL, NULL);
	if (st->error == NULL){
            Py_DECREF(moose_module);
            INITERROR;
	}
        int registered = Py_AtExit(&finalize);
        if (registered != 0){
            cerr << "Failed to register finalize() to be called at exit. " << endl;
        }

        import_array();
        // Add Id type
        // Py_TYPE(&IdType) = &PyType_Type; // unnecessary - filled in by PyType_Ready
        IdType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&IdType) < 0){
            PyErr_Print();
            exit(-1);
        };            
        Py_INCREF(&IdType);
        PyModule_AddObject(moose_module, "ematrix", (PyObject*)&IdType);

        // Add ObjId type
        // Py_TYPE(&ObjIdType) = &PyType_Type; // unnecessary - filled in by PyType_Ready
        ObjIdType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&ObjIdType) < 0){
            PyErr_Print();
            exit(-1);
        };
        Py_INCREF(&ObjIdType);
        PyModule_AddObject(moose_module, "melement", (PyObject*)&ObjIdType);

        // Add LookupField type
        // Py_TYPE(&moose_LookupField) = &PyType_Type;  // unnecessary - filled in by PyType_Ready        
        // moose_LookupField.tp_new = PyType_GenericNew;
        if (PyType_Ready(&moose_LookupField) < 0){
            PyErr_Print();
            exit(-1);
        }        
        Py_INCREF(&moose_LookupField);
        PyModule_AddObject(moose_module, "LookupField", (PyObject*)&moose_ElementField);

        if (PyType_Ready(&moose_ElementField) < 0){
            PyErr_Print();
            exit(-1);
        }        
        Py_INCREF(&moose_ElementField);
        PyModule_AddObject(moose_module, "ElementField", (PyObject*)&moose_ElementField);
        // Add DestField type
        // Py_TYPE(&moose_DestField) = &PyType_Type; // unnecessary - filled in by PyType_Ready
        // moose_DestField.tp_flags = Py_TPFLAGS_DEFAULT;
        // moose_DestField.tp_call = moose_DestField_call;
        // moose_DestField.tp_doc = DestField_documentation;
        // moose_DestField.tp_new = PyType_GenericNew;
        if (PyType_Ready(&moose_DestField) < 0){
            PyErr_Print();
            exit(-1);
        }
        Py_INCREF(&moose_DestField);
        PyModule_AddObject(moose_module, "DestField", (PyObject*)&moose_DestField);
        
        // PyModule_AddIntConstant(moose_module, "SINGLETHREADED", isSingleThreaded);
        PyModule_AddIntConstant(moose_module, "NUMCORES", numCores);
        PyModule_AddIntConstant(moose_module, "NUMNODES", numNodes);
        // PyModule_AddIntConstant(moose_module, "NUMPTHREADS", numProcessThreads);
        PyModule_AddIntConstant(moose_module, "MYNODE", myNode);
        PyModule_AddIntConstant(moose_module, "INFINITE", isInfinite);
        PyModule_AddStringConstant(moose_module, "__version__", SHELLPTR->doVersion().c_str());
        PyModule_AddStringConstant(moose_module, "VERSION", SHELLPTR->doVersion().c_str());
        PyModule_AddStringConstant(moose_module, "SVN_REVISION", SHELLPTR->doRevision().c_str());
        PyObject * module_dict = PyModule_GetDict(moose_module);
        clock_t defclasses_start = clock();
        if (!defineAllClasses(module_dict)){
            PyErr_Print();
            exit(-1);
        }
        for (map <string, PyTypeObject * >::iterator ii = get_moose_classes().begin();
             ii != get_moose_classes().end(); ++ii){
            PyModule_AddObject(moose_module, ii->first.c_str(), (PyObject*)(ii->second));
        }
             
        clock_t defclasses_end = clock();
        cout << "Info: Time to define moose classes:" << (defclasses_end - defclasses_start) * 1.0 /CLOCKS_PER_SEC << endl;
        PyGILState_Release(gstate);
        clock_t modinit_end = clock();
        cout << "Info: Time to initialize module:" << (modinit_end - modinit_start) * 1.0 /CLOCKS_PER_SEC << endl;
        if (doUnitTests){
            test_moosemodule();
        }
#ifdef PY3K
	return moose_module;
#endif
    } //! init_moose

} // end extern "C"

//////////////////////////////////////////////
// Main function
//////////////////////////////////////////////

int main(int argc, char* argv[])
{
#ifdef PY3K
    size_t len = strlen(argv[0]);
    wchar_t * warg = (wchar_t*)calloc(sizeof(wchar_t), len);
    mbstowcs(warg, argv[0], len);
#else
    char * warg = argv[0];
#endif
    for (int ii = 0; ii < argc; ++ii){
    cout << "ARGV: " << argv[ii];
}
    cout << endl;
    Py_SetProgramName(warg);
    Py_Initialize();
    MODINIT(_moose);
#if PY3K
    free(warg);
#endif
    return 0;
}

// 
// moosemodule.cpp ends here
