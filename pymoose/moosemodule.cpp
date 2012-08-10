// moosemodule.cpp --- 
// 
// Filename: moosemodule.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Thu Mar 10 11:26:00 2011 (+0530)
// Version: 
// Last-Updated: Fri Aug 10 17:18:04 2012 (+0530)
//           By: subha
//     Update #: 9261
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
#include "../basecode/DataId.h"
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
extern Id init(int argc, char ** argv, bool& doUnitTests, bool& doRegressionTests);

extern void initMsgManagers();
extern void destroyMsgManagers();
extern void speedTestMultiNodeIntFireNetwork( 
	unsigned int size, unsigned int runsteps );
#ifdef DO_UNIT_TESTS
extern void regressionTests();
#endif
extern bool benchmarkTests( int argc, char** argv );
extern int getNumCores();

// C-wrapper to be used by Python
extern "C" {
    /////////////////////////////////////////////////////////////////
    // Module globals 
    /////////////////////////////////////////////////////////////////
    static int verbosity = 1;
    static int isSingleThreaded = 0;
    static int isInfinite = 0;
    static int numNodes = 1;
    static int numCores = 1;
    static int myNode = 0;
    static int numProcessThreads = 0;
    static int doUnitTests = 0;
    static int doRegressionTests = 0;
    static int quitFlag = 0;
    static Id shellId;
    static PyObject * MooseError;
    static PyObject* get_Id_attr(_Id * id, string attribute)
    {
        if (attribute == "path"){
            return moose_Id_getPath(id);
        } else if (attribute == "value"){
            return moose_Id_getValue(id);
        } else if (attribute == "shape"){
            return moose_Id_getShape(id);
        } else if (attribute == "class_"){ // class is a common
                                          // attribute to all ObjIds
                                          // under this Id. Expect it
                                          // to be a single value in
                                          // stead of a list of class
                                          // names.
            string classname = Field<string>::get(id->id_, "class");
            return Py_BuildValue("s", classname.c_str());
        }
        return NULL;
    }

    static PyObject * get_ObjId_attr(_ObjId * oid, string attribute)
    {
        if (attribute == "id_"){
            return moose_ObjId_getId(oid);
        } else if (attribute == "dindex"){
            return moose_ObjId_getDataIndex(oid);
        } else if (attribute == "findex"){
            return moose_ObjId_getFieldIndex(oid);
        }
        return NULL;
    }
                
    // Global store of defined MOOSE classes.
    static map<string, PyTypeObject *>& get_moose_classes()
    {
        static map<string, PyTypeObject *> defined_classes;
        return defined_classes;
    }
    
    // Global storage for PyGetSetDef structs for LookupFields.
    static map<string, vector <PyGetSetDef> >& get_getsetdefs()
    {
        static map<string, vector <PyGetSetDef>  > getset_defs;
        return getset_defs;
    }
    
    // Global storage for every LookupField we create.
    static map<string, PyObject *>& get_inited_lookupfields()
    {
        static map<string, PyObject *> inited_lookupfields;
        return inited_lookupfields;
    }
    
    static map< string, PyObject * >& get_inited_destfields()
    {
        static map<string, PyObject * > inited_destfields;
        return inited_destfields;
    }

    static map< string, PyObject *>& get_inited_elementfields()
    {
        static map< string, PyObject *> inited_elementfields;
        return inited_elementfields;
    }
    
    /**
       map of fields which are aliased in Python to avoid collision
       with Python keywords.
     */
    static const map<string, string>& get_field_alias()
    {
        static map<string, string> alias;
        if (alias.empty()){
            alias["class_"] = "class";
            alias["lambda_"] = "lambda";
        }
        return alias;
    }


    // Minimum number of arguments for setting destFinfo - 1-st
    // the finfo name.
    static Py_ssize_t minArgs = 1;
    
    // Arbitrarily setting maximum on variable argument list. Read:
    // http://www.swig.org/Doc1.3/Varargs.html to understand why
    static Py_ssize_t maxArgs = 10;

    
    ///////////////////////////////////
    // Python datatype checking macros
    ///////////////////////////////////
    
#define Id_Check(v) (Py_TYPE(v) == &IdType)
#define Id_SubtypeCheck(v) (PyType_IsSubtype(Py_TYPE(v),&IdType))
#define ObjId_Check(v) (Py_TYPE(v) == &ObjIdType)
#define ObjId_SubtypeCheck(v) (PyType_IsSubtype(Py_TYPE(v), &ObjIdType))
#define LookupField_Check(v) (Py_TYPE(v) == &LookupFieldType)
    
    // Macro to create the Shell * out of shellId
#define ShellPtr reinterpret_cast<Shell*>(shellId.eref().data())    

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
        it = argmap.find("SINGLETHREADED");
        if (it != argmap.end()){
            istringstream(it->second) >> isSingleThreaded;
            if (isSingleThreaded){
                args.push_back("-s");
            }
        }
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
        it = argmap.find("NUMPTHREADS");
        if (it != argmap.end()){
            istringstream(it->second) >> numProcessThreads;
            args.push_back("-t");
            args.push_back(it->second);	            
        }
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
        it = argmap.find("DOUNITTESTS");
        if (it != argmap.end()){
            istringstream(it->second) >> doUnitTests;            
        }
        it = argmap.find("DOREGRESSIONTESTS");
        if (it != argmap.end()){
            istringstream(it->second) >> doRegressionTests;            
        }
        
        if (verbosity > 0){
            cout << "ENVIRONMENT: " << endl
                 << "----------------------------------------" << endl
                 << "   SINGLETHREADED = " << isSingleThreaded << endl
                 << "   INFINITE = " << isInfinite << endl
                 << "   NUMNODES = " << numNodes << endl
                 << "   NUMPTHREADS = " << numProcessThreads << endl
                 << "   VERBOSITY = " << verbosity << endl
                 << "   DOUNITTESTS = " << doUnitTests << endl
                 << "   DOREGRESSIONTESTS = " << doRegressionTests << endl
                 << "========================================" << endl;
        }
        return args;
    } //! setup_runtime_env()

    /**
       Create the shell instance. This calls
       basecode/main.cpp:init(argc, argv) to do the initialization.
    */
    void create_shell(int argc, char ** argv)
    {
        static int inited = 0;
        if (inited){
            cout << "Warning: shell already initialized." << endl;
            return;
        }
        bool dounit = doUnitTests != 0;
        bool doregress = doRegressionTests != 0;
        // Utilize the main::init function which has friend access to Id
        shellId = init(argc, argv, dounit, doregress);
        Qinfo::initMutex(); // Mutex used to align Parser and MOOSE threads        
        if (dounit){
            nonMpiTests( ShellPtr ); // These tests do not need the process loop.
        }

        if (!ShellPtr->isSingleThreaded()){
            ShellPtr->launchThreads();
        }
        if ( ShellPtr->myNode() == 0 ) {
            if (dounit){
                mpiTests();
                processTests( ShellPtr );
                regressionTests();
            }
            if ( benchmarkTests( argc, argv ) || quitFlag ){
                ShellPtr->doQuit();
            }
        }        
    } //! create_shell()

    /**
       Clean up after yourself.
    */
    void finalize()
    {
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
        ShellPtr->doQuit();
        // Cleanup threads
        if (!ShellPtr->isSingleThreaded()){
            ShellPtr->joinThreads();
            Qinfo::freeMutex();
        }
        // Destroy the Shell object
        Neutral* ns = reinterpret_cast<Neutral*>(shellId.element()->dataHandler()->data(0));
        ns->destroy( shellId.eref(), 0, 0);
#ifdef USE_MPI
        MPI_Finalize();
#endif
    } //! finalize()


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
        unsigned int numFinfos = Field<unsigned int>::get(ObjId(classId, 0), "num_" + finfoType);
        Id fieldId(classId.path() + "/" + finfoType);
        for (unsigned int ii = 0; ii < numFinfos; ++ii){
            string _fieldName = Field<string>::get(ObjId(fieldId, DataId(0, ii, 0)), "name");
            if (fieldName == _fieldName){                
                fieldType = Field<string>::get(ObjId(fieldId, DataId(0, ii, 0)), "type");
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
    // Does not get called at all by PyObject_New. See:
    // http://www.velocityreviews.com/forums/t344033-pyobject_new-not-running-tp_new-for-iterators.html
    // static PyObject * moose_Field_new(PyTypeObject *type,
    //                                   PyObject *args, PyObject *kwds)
    // {
    //     _Field *self = NULL;
    //     self = (_Field*)type->tp_alloc(type, 0);
    //     if (self != NULL){            
    //         self->name = NULL;
    //         self->owner = ObjId::bad;
    //     }        
    //     return (PyObject*)self;
    // }
    /**
       Initialize field with ObjId and fieldName.
    */
    static int moose_Field_init(_Field * self, PyObject * args, PyObject * kwargs)
    {
        extern PyTypeObject ObjIdType;
        PyObject * owner;
        char * fieldName;
        if (!PyArg_ParseTuple(args, "Os:moose_Field_init", &owner, &fieldName)){
            return -1;
        }
        if (fieldName == NULL){
            PyErr_SetString(PyExc_ValueError, "fieldName cannot be NULL");
            return -1;
        }
        if (owner == NULL){
            PyErr_SetString(PyExc_ValueError, "owner cannot be NULL");
            return -1;
        }
        if (!ObjId_SubtypeCheck(owner)){
            PyErr_SetString(PyExc_TypeError, "Owner must be subtype of ObjId");
            return -1;
        }
        self->owner = ((_ObjId*)owner)->oid_;
        size_t size = strlen(fieldName);
        char * name = (char*)calloc(size+1, sizeof(char));
        strncpy(name, fieldName, size);
        self->name = name;
        // In earlier version I tried to deallocate the existing
        // self->name if it is not NULL. But it turns out that it
        // causes a SIGABRT. In any case it should not be an issue as
        // we can safely assume __init__ will be called only once in
        // this case. The Fields are created only internally at
        // initialization of the MOOSE module.
        return 0;
    }

    /// Return the hash of the string `{objectpath}.{fieldName}`
    static long moose_Field_hash(_Field * self)
    {
        string fieldPath = self->owner.path() + "." + self->name;
        PyObject * path = PyString_FromString(fieldPath.c_str());
        long hash = PyObject_Hash(path);
        Py_XDECREF(path);
        return hash;    
    }

    /// String representation of fields is `{objectpath}.{fieldName}`
    static PyObject * moose_Field_repr(_Field * self)
    {
        ostringstream fieldPath;
        fieldPath << self->owner.path() << "." << self->name;
        return PyString_FromString(fieldPath.str().c_str());
    }

    
    PyDoc_STRVAR(moose_Field_documentation,
                 "Base class for MOOSE fields.\n"
                 "\n"
                 "Instances contain the field name and a pointer to the owner object.");


    static PyTypeObject moose_Field = {
        PyObject_HEAD_INIT(NULL)
        0,                                              /* tp_internal */
        "moose.Field",                                  /* tp_name */
        sizeof(_Field),                                 /* tp_basicsize */
        0,                                              /* tp_itemsize */
        0,// (destructor)moose_Field_dealloc,                /* tp_dealloc */
        0,                                              /* tp_print */
        0,                                              /* tp_getattr */
        0,                                              /* tp_setattr */
        0,                                              /* tp_compare */
        (reprfunc)moose_Field_repr,                     /* tp_repr */
        0,                                              /* tp_as_number */
        0,                                              /* tp_as_sequence */
        0,                                              /* tp_as_mapping */
        (hashfunc)moose_Field_hash,                     /* tp_hash */
        0,                                              /* tp_call */
        (reprfunc)moose_Field_repr,                     /* tp_str */
        0,                  /* tp_getattro */
        PyObject_GenericSetAttr,                        /* tp_setattro */
        0,                                              /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        moose_Field_documentation,
        0,                                              /* tp_traverse */
        0,                                              /* tp_clear */
        0,                                              /* tp_richcompare */
        0,                                              /* tp_weaklistoffset */
        0,                                              /* tp_iter */
        0,                                              /* tp_iternext */
        0,                                              /* tp_methods */
        0,                                              /* tp_members */
        0,                                              /* tp_getset */
        0,                                              /* tp_base */
        0,                                              /* tp_dict */
        0,                                              /* tp_descr_get */
        0,                                              /* tp_descr_set */
        0,                                              /* tp_dictoffset */
        (initproc)moose_Field_init,                     /* tp_init */
        0,                                              /* tp_alloc */
        0,                                              /* tp_new */
        0,                                              /* tp_free */
    };

    static PyObject * moose_LookupField_getItem(_Field * self, PyObject * key)
    {
        return getLookupField(self->owner, self->name, key);
    }

    static int moose_LookupField_setItem(_Field * self, PyObject * key,
                                            PyObject * value)
    {
        return setLookupField(self->owner,
                              self->name, key, value);
    }

    /**
       The mapping methods make it act like a Python dictionary.
    */
    static PyMappingMethods LookupFieldMappingMethods = {
        0,
        (binaryfunc)moose_LookupField_getItem,
        (objobjargproc)moose_LookupField_setItem,
    };

    PyDoc_STRVAR(moose_LookupField_documentation,
                 "LookupField is dictionary-like fields that map keys to values.\n"
                 "The keys need not be fixed, as in case of interpolation tables,\n"
                 "keys can be any number and the corresponding value is dynamically\n"
                 "computed by the method of interpolation.\n"
                 "Use moose.doc('classname.fieldname') to display builtin\n"
                 "documentation for `field` in class `classname`.\n");
    static PyTypeObject moose_LookupField = {
        PyObject_HEAD_INIT(NULL)
        0,                                              /* tp_internal */
        "moose.LookupField",                                  /* tp_name */
        sizeof(_Field),                                 /* tp_basicsize */
        0,                                              /* tp_itemsize */
        0,                /* tp_dealloc */
        0,                                              /* tp_print */
        0,                                              /* tp_getattr */
        0,                                              /* tp_setattr */
        0,                                              /* tp_compare */
        (reprfunc)moose_Field_repr,                     /* tp_repr */
        0,                                              /* tp_as_number */
        0,                                              /* tp_as_sequence */
        &LookupFieldMappingMethods,                      /* tp_as_mapping */
        (hashfunc)moose_Field_hash,                     /* tp_hash */
        0,                                              /* tp_call */
        (reprfunc)moose_Field_repr,                     /* tp_str */
        0,                  /* tp_getattro */
        PyObject_GenericSetAttr,                        /* tp_setattro */
        0,                                              /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,
        moose_LookupField_documentation,
        0,                                              /* tp_traverse */
        0,                                              /* tp_clear */
        0,                                              /* tp_richcompare */
        0,                                              /* tp_weaklistoffset */
        0,                                              /* tp_iter */
        0,                                              /* tp_iternext */
        0,                                              /* tp_methods */
        0,                                              /* tp_members */
        0,                                              /* tp_getset */
        &moose_Field,                                              /* tp_base */
        0,                                              /* tp_dict */
        0,                                              /* tp_descr_get */
        0,                                              /* tp_descr_set */
        0,                                              /* tp_dictoffset */
        (initproc)moose_Field_init,                     /* tp_init */
        0,                                              /* tp_alloc */
        0,                       /* tp_new */
        0,                                              /* tp_free */
    };


    static PyObject * moose_DestField_call(PyObject * self, PyObject * args,
                                           PyObject * kw)
    {
        // We copy the name as the first argument into a new argument tuple. 
        PyObject * newargs = PyTuple_New(PyTuple_Size(args)+1); // one extra for the field name
        PyObject * name = PyString_FromString(((_Field*)self)->name);
        if (name == NULL){
            Py_XDECREF(newargs);
            return NULL;
        }
        if (PyTuple_SetItem(newargs, 0, name) != 0){
            Py_XDECREF(newargs);
            return NULL;
        }
        // We copy the arguments in `args` into the new argument tuple
        Py_ssize_t argc =  PyTuple_Size(args);
        for (Py_ssize_t ii = 0; ii < argc; ++ii){
            PyObject * arg = PyTuple_GetItem(args, ii);
            if (arg != NULL){
                PyTuple_SetItem(newargs, ii+1, arg);
            } else {
                Py_XDECREF(newargs);
                return NULL;
            }
        }
        // Call ObjId._setDestField with the new arguments
        return _setDestField(((_Field*)self)->owner,
                                        newargs);
    }

    PyDoc_STRVAR(moose_DestField_documentation,
                 "DestField is a method field, i.e. it can be called like a function.\n"
                 "Use moose.doc('classname.fieldname') to display builtin\n"
                 "documentation for `field` in class `classname`.\n");
                 

    static PyTypeObject moose_DestField = {
        PyObject_HEAD_INIT(NULL)
        0,                                              /* tp_internal */
        "moose.DestField",                              /* tp_name */
        sizeof(_Field),                                 /* tp_basicsize */
        0,                                              /* tp_itemsize */
        0,                /* tp_dealloc */
        0,                                              /* tp_print */
        0,                                              /* tp_getattr */
        0,                                              /* tp_setattr */
        0,                                              /* tp_compare */
        (reprfunc)moose_Field_repr,                     /* tp_repr */
        0,                                              /* tp_as_number */
        0,                                              /* tp_as_sequence */
        0,                                              /* tp_as_mapping */
        (hashfunc)moose_Field_hash,                     /* tp_hash */
        moose_DestField_call,                           /* tp_call */
        (reprfunc)moose_Field_repr,                     /* tp_str */
        0,                                              /* tp_getattro */
        PyObject_GenericSetAttr,                        /* tp_setattro */
        0,                                              /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,
        moose_DestField_documentation,
        0,                                              /* tp_traverse */
        0,                                              /* tp_clear */
        0,                                              /* tp_richcompare */
        0,                                              /* tp_weaklistoffset */
        0,                                              /* tp_iter */
        0,                                              /* tp_iternext */
        0,                                              /* tp_methods */
        0,                                              /* tp_members */
        0,                                              /* tp_getset */
        &moose_Field,                                              /* tp_base */
        0,                                              /* tp_dict */
        0,                                              /* tp_descr_get */
        0,                                              /* tp_descr_set */
        0,                                              /* tp_dictoffset */
        (initproc)moose_Field_init,                     /* tp_init */
        0,                                              /* tp_alloc */
        0,                       /* tp_new */
        0,                                              /* tp_free */
    };

    PyDoc_STRVAR(moose_ElementField_documentation,
                 "ElementField represents fields that are themselves elements. For\n"
                 "example, synapse in an IntFire neuron.\n");

    PyDoc_STRVAR(moose_ElementField_num_documentation,
                 "Return number of entries in the field.");
    static char numfield[] = "num";
    static PyGetSetDef ElementFieldGetSetters[] = {
        {numfield,
         (getter)moose_ElementField_getNum,
         (setter)moose_ElementField_setNum,
         moose_ElementField_num_documentation,
         NULL},
        {NULL}, /* sentinel */
    };
    
    static PySequenceMethods ElementFieldSequenceMethods = {
        (lenfunc)moose_ElementField_getNum, // sq_length
        0, //sq_concat
        0, //sq_repeat
        (ssizeargfunc)moose_ElementField_getItem, //sq_item
        0, // getslice
        0, //sq_ass_item
        0, // setslice
        0, // sq_contains
        0, // sq_inplace_concat
        0 // sq_inplace_repeat
    };
    
    static PyTypeObject moose_ElementField = {
        PyObject_HEAD_INIT(NULL)
        0,                                              /* tp_internal */
        "moose.ElementField",                              /* tp_name */
        sizeof(_Field),                                 /* tp_basicsize */
        0,                                              /* tp_itemsize */
        0,                                              /* tp_dealloc */
        0,                                              /* tp_print */
        0,                                              /* tp_getattr */
        0,                                              /* tp_setattr */
        0,                                              /* tp_compare */
        (reprfunc)moose_Field_repr,                     /* tp_repr */
        0,                                              /* tp_as_number */
        &ElementFieldSequenceMethods,                   /* tp_as_sequence */
        0,                                              /* tp_as_mapping */
        (hashfunc)moose_Field_hash,                     /* tp_hash */
        0,                                              /* tp_call */
        (reprfunc)moose_Field_repr,                     /* tp_str */
        0,                                              /* tp_getattro */
        PyObject_GenericSetAttr,                        /* tp_setattro */
        0,                                              /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT,
        moose_ElementField_documentation,
        0,                                              /* tp_traverse */
        0,                                              /* tp_clear */
        0,                                              /* tp_richcompare */
        0,                                              /* tp_weaklistoffset */
        0,                                              /* tp_iter */
        0,                                              /* tp_iternext */
        0,                           /* tp_methods */
        0,                                              /* tp_members */
        ElementFieldGetSetters,                                              /* tp_getset */
        &moose_Field,                                              /* tp_base */
        0,                                              /* tp_dict */
        0,                                              /* tp_descr_get */
        0,                                              /* tp_descr_set */
        0,                                              /* tp_dictoffset */
        (initproc)moose_Field_init,                     /* tp_init */
        0,                                              /* tp_alloc */
        0,                       /* tp_new */
        0,                                              /* tp_free */        
    };

    ///////////////////////////////////////////////
    // Python method lists for PyObject of Id
    ///////////////////////////////////////////////
    
    static PyMethodDef IdMethods[] = {
        // {"init", (PyCFunction)moose_Id_init, METH_VARARGS,
        //  "Initialize a Id object."},
        {"delete", (PyCFunction)moose_Id_delete, METH_VARARGS,
         "Delete the underlying moose element"},
        {"getValue", (PyCFunction)moose_Id_getValue, METH_NOARGS,
         "Return integer representation of the id of the element."},
        {"getPath", (PyCFunction)moose_Id_getPath, METH_NOARGS,
         "Return the path of this Id object."},
        {"getShape", (PyCFunction)moose_Id_getShape, METH_NOARGS,
         "Get the shape of the Id object as a tuple."},
        {"setField", (PyCFunction)moose_Id_setField, METH_VARARGS,
         "setField(fieldname, value_vector)\n"
         "\n"
         "Set the value of `fieldname` in all elements under this Id.\n"
         "\n"
         "Parameters\n"
         "----------\n"
         "fieldname: str\n"
         "\tfield to be set.\n"
         "value: sequence of values\n"
         "\tsequence of values corresponding to individual elements under this\n"
         "Id.\n"
         "\n"
         "NOTE: This is an interface to SetGet::setVec\n"
        },
        {NULL, NULL, 0, NULL},        /* Sentinel */        
    };

    static PySequenceMethods IdSequenceMethods = {
        (lenfunc)moose_Id_getLength, // sq_length
        0, //sq_concat
        0, //sq_repeat
        (ssizeargfunc)moose_Id_getItem, //sq_item
        (ssizessizeargfunc)moose_Id_getSlice, // getslice
        0, //sq_ass_item
        0, // setslice
        (objobjproc)moose_Id_contains, // sq_contains
        0, // sq_inplace_concat
        0 // sq_inplace_repeat
    };

    
    ///////////////////////////////////////////////
    // Type defs for PyObject of Id
    ///////////////////////////////////////////////
    PyTypeObject IdType = { 
        PyObject_HEAD_INIT(NULL)               /* tp_head */
        0,                                  /* tp_internal */
        "moose.Id",                  /* tp_name */
        sizeof(_Id),                    /* tp_basicsize */
        0,                                  /* tp_itemsize */
        0,                    /* tp_dealloc */
        0,                                  /* tp_print */
        0,                                  /* tp_getattr */
        0,                                  /* tp_setattr */
        0,                                  /* tp_compare */
        (reprfunc)moose_Id_repr,                        /* tp_repr */
        0,                                  /* tp_as_number */
        &IdSequenceMethods,             /* tp_as_sequence */
        0,                                  /* tp_as_mapping */
        (hashfunc)moose_Id_hash,                                  /* tp_hash */
        0,                                  /* tp_call */
        (reprfunc)moose_Id_str,               /* tp_str */
        (getattrofunc)moose_Id_getattro,            /* tp_getattro */
        (setattrofunc)moose_Id_setattro,            /* tp_setattro */
        0,                                  /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        "Id object of moose. Which can act as an array object.",
        0,                                  /* tp_traverse */
        0,                                  /* tp_clear */
        (richcmpfunc)moose_Id_richCompare,       /* tp_richcompare */
        0,                                  /* tp_weaklistoffset */
        0,                                  /* tp_iter */
        0,                                  /* tp_iternext */
        IdMethods,                     /* tp_methods */
        0,                    /* tp_members */
        0,                                  /* tp_getset */
        0,                                 /* tp_base */
        0,                                  /* tp_dict */
        0,                                  /* tp_descr_get */
        0,                                  /* tp_descr_set */
        0,                                  /* tp_dictoffset */
        (initproc) moose_Id_init,   /* tp_init */
        0,                /* tp_alloc */
        0,                  /* tp_new */
        0,                      /* tp_free */
    };

    


    //////////////////////////////////////////////////
    // Id functions
    //////////////////////////////////////////////////

    static int moose_Id_create(_Id * self, string path, PyObject * dims,
                               char * type)
    {
        if (type == NULL){
            PyErr_SetString(PyExc_ValueError,
                            "moose_Id_create: type cannot be NULL.");
            return -1;
        }        
        string trimmed_type = trim(string(type));
        if (trimmed_type.length() <= 0){
            PyErr_SetString(PyExc_ValueError,
                            "moose_Id_create: type must be non-empty string.");
            return -1;
        }        
        string trimmed_path(path);
        size_t length = trimmed_path.length();
        
        //  paths ending with '/' should raise exception
        if ((length > 1) && (trimmed_path[length - 1] == '/')){
            PyErr_SetString(PyExc_ValueError,
                            "moose_Id_create: Non-root path must not end with '/'");
            return -1;
        }
        // prepare the dimensions vector
        vector <int> vec_dims;
        Py_ssize_t num_dims = 1;
        if (dims){
            // First try to use it as a tuple of dimensions
            if (PySequence_Check(dims)){
                num_dims = PySequence_Length(dims);
                for (Py_ssize_t ii = 0; ii < num_dims; ++ ii){
                    PyObject* dim = PySequence_GetItem(dims, ii);
                    long dim_value = PyInt_AsLong(dim);
                    Py_XDECREF(dim);                    
                    if ((dim_value == -1) && PyErr_Occurred()){
                        return -1;
                    }
                    vec_dims.push_back((unsigned int)dim_value);
                }
            } else if (PyInt_Check(dims)){ // 1D array
                num_dims = PyInt_AsLong(dims);
                if (num_dims <= 0){
                    num_dims = 1;
                }
            }
        }        
        if (vec_dims.empty()){
            vec_dims.push_back(num_dims);
        }
        // object does not exist, create new
        string parent_path;
        string name;
        size_t pos = trimmed_path.rfind("/");
        if (pos != string::npos){
            name = trimmed_path.substr(pos+1);
            parent_path = trimmed_path.substr(0, pos);
        } else {
            name = trimmed_path;
        }
        // handle relative path
        if (trimmed_path[0] != '/'){
            string current_path = ShellPtr->getCwe().path();
            if (current_path != "/"){
                parent_path = current_path + "/" + parent_path;
            } else {
                parent_path = current_path + parent_path;
            }
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
        self->id_ = ShellPtr->doCreate(string(type),
                                       parent_id,
                                       string(name),
                                       vector<int>(vec_dims));
         
        return 0;            
    }
    
    static int moose_Id_init_from_path(_Id * self, PyObject * args, PyObject * kwargs)
    {
        char _path[] = "path";
        char _dtype[] = "dtype";
        char _dims[] = "dims";
        static char * kwlist[] = {_path, _dims, _dtype, NULL};
        char * path = NULL;
        char _default_type[] = "Neutral";
        char *type = _default_type;
        PyObject * dims = NULL;
        // If argument parsing fails, flag taht by returning
        // -1. Caller should try alternative parsing.
        if (kwargs == NULL){
            if( !PyArg_ParseTuple(args,
                                                "s|Os:moose_Id_init",
                                                &path,
                                                &dims,
                                                &type)){
            return -1;
            }
            PyErr_Clear();
        } else if (!PyArg_ParseTupleAndKeywords(args,
                                                kwargs,
                                                "s|Os:moose_Id_init",
                                                kwlist,
                                                &path,
                                                &dims,
                                                &type)){
            return -1;
        }
        // Parsing args successful, if any error happens now,
        // different argument processing will not help. Indicate that
        // by returning -2
        string trimmed_path(path);
        trimmed_path = trim(trimmed_path);
        size_t length = trimmed_path.length();
        if (length <= 0){
            PyErr_SetString(PyExc_ValueError,
                            "path must be non-empty string.");
            return -2;
        }
        self->id_ = Id(trimmed_path);
        // Return already existing object
        if (self->id_ != Id() ||
            trimmed_path == "/" ||
            trimmed_path == "/root"){
            return 0;
        }
        if (moose_Id_create(self, trimmed_path, dims,
                             type) == 0){
            return 0;
        }
        return -2;
    }
    
    static int moose_Id_init(_Id * self, PyObject * args, PyObject * kwargs)
    {
        extern PyTypeObject IdType;
        PyObject * src = NULL;
        unsigned int id = 0;
        int ret = moose_Id_init_from_path(self, args, kwargs);
        if (ret == 0){
            return 0;
        } else if (ret < -1){
            return ret;
        }
        PyErr_Clear();
        if (PyArg_ParseTuple(args, "I:moose_Id_init", &id)){
            self->id_ = Id(id);
            return 0;
        }
        PyErr_Clear();        
        if (PyArg_ParseTuple(args, "O:moose_Id_init", &src) && Id_Check(src)){
            self->id_ = ((_Id*)src)->id_;
            return 0;
        }
        return -1;
    }// ! moose_Id_init

    static long moose_Id_hash(_Id * self)
    {
        return self->id_.value(); // hash is the same as the Id value
    }

    
    // 2011-03-23 15:14:11 (+0530)
    // 2011-03-26 17:02:19 (+0530)
    //
    // 2011-03-26 19:14:34 (+0530) - This IS UGLY! Destroying one
    // ObjId will destroy the containing element and invalidate all
    // the other ObjId with the same Id.
    // 2011-03-28 13:44:49 (+0530)
    static PyObject * moose_Id_delete(_Id * self, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":moose_Id_delete")){
            return NULL;
        }
        if (self->id_ == Id()){
            PyErr_SetString(PyExc_ValueError, "Cannot delete moose shell.");
            return NULL;
        }
        ShellPtr->doDelete(self->id_);
        self->id_ = Id();
        Py_RETURN_NONE;
    }
    static PyObject * moose_Id_repr(_Id * self)
    {
        return PyString_FromFormat("<Id: id=%u, path=%s>", self->id_.value(), self->id_.path().c_str());
    } // !  moose_Id_repr
    static PyObject * moose_Id_str(_Id * self)
    {
        return PyString_FromFormat("%s", Id::id2str(self->id_).c_str());
    } // !  moose_Id_str

    // 2011-03-23 15:09:19 (+0530)
    static PyObject* moose_Id_getValue(_Id * self)
    {
        unsigned int id = self->id_.value();        
        PyObject * ret = Py_BuildValue("I", id);
        return ret;
    }
    /**
       Not to be redone. 2011-03-23 14:42:48 (+0530)
    */
    static PyObject * moose_Id_getPath(_Id * self)
    {
        string path = self->id_.path();
        PyObject * ret = Py_BuildValue("s", path.c_str());
        return ret;
    }

    /** Subset of sequence protocol functions */
    static Py_ssize_t moose_Id_getLength(_Id * self)
    {
        vector< unsigned int> dims = Field< vector <unsigned int> >::get(ObjId(self->id_), "objectDimensions");
        if (dims.empty()){
            return (Py_ssize_t)1; // this is a bug in basecode - dimension 1 is returned as an empty vector
        } else {
            return (Py_ssize_t)dims[0];
        }
    }
    static PyObject * moose_Id_getShape(_Id * self)
    {
        vector< unsigned int> dims = Field< vector <unsigned int> >::get(self->id_, "objectDimensions");
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
    static PyObject * moose_Id_getItem(_Id * self, Py_ssize_t index)
    {
        extern PyTypeObject ObjIdType;
        if (index < 0){
            index += moose_Id_getLength(self);
        }
        if ((index < 0) || (index >= moose_Id_getLength(self))){
            PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
            return NULL;
        }
        _ObjId * ret = PyObject_New(_ObjId, &ObjIdType);
        ret->oid_ = ObjId(self->id_, index);
        return (PyObject*)ret;
    }
    static PyObject * moose_Id_getSlice(_Id * self, PyObject * args)
    {
        extern PyTypeObject ObjIdType;
        Py_ssize_t start, end;
        if (!PyArg_ParseTuple(args, "ii:moose_Id_getSlice", &start, &end)){
            return NULL;
        }
        Py_ssize_t len = moose_Id_getLength(self);
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
                Py_XDECREF(value);
                return NULL;
            }
        }
        return ret;
    }
    
    static PyObject * moose_Id_richCompare(_Id * self, PyObject * other, int op)
    {
        extern PyTypeObject IdType;
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
    
    static int moose_Id_contains(_Id * self, PyObject * obj)
    {
        extern PyTypeObject ObjIdType;
        int ret = 0;
        if (ObjId_Check(obj)){
            ret = (((_ObjId*)obj)->oid_.id == self->id_);
        }
        return ret;
    }
    
    static PyObject * moose_Id_getattro(_Id * self, PyObject * attr)
    {
        char * field = PyString_AsString(attr);
        PyObject * _ret = get_Id_attr(self, field);
        if (_ret != NULL){
            return _ret;
        }        
        string class_name = Field<string>::get(self->id_, "class");
        string type = getFieldType(class_name, string(field), "valueFinfo");
        if (type.empty()){
            // Check if this field name is aliased and update fieldname and type if so.
            map<string, string>::const_iterator it = get_field_alias().find(string(field));
            if (it != get_field_alias().end()){
                field = const_cast<char*>((it->second).c_str());
                type = getFieldType(Field<string>::get(self->id_, "class"), it->second, "valueFinfo");
                // Update attr for next level (PyObject_GenericGetAttr) in case.
                Py_XDECREF(attr);
                attr = PyString_FromString(field);
            }
        }
        if (type.empty()){
            return PyObject_GenericGetAttr((PyObject*)self, attr);            
        }
        char ftype = shortType(type);
        if (!ftype){
            return PyObject_GenericGetAttr((PyObject*)self, attr);
        }

#define ID_GETVEC(TYPE)                                                 \
        {                                                               \
            vector<TYPE> val;                                           \
            Field< TYPE >::getVec(self->id_, string(field), val);       \
            npy_intp dims = val.size();                                 \
            PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &dims, get_npy_typenum(typeid(TYPE))); \
            char * ptr = PyArray_BYTES(ret);                            \
            memcpy(ptr, &val[0], val.size() * sizeof(TYPE));            \
            return PyArray_Return(ret);                                 \
        } // ID_GETVEC
        
        switch (ftype){
            case 'b':         {                                                               
                vector<bool> val;                                           
                Field< bool >::getVec(self->id_, string(field), val);
                npy_intp dims = val.size();
                PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &dims, get_npy_typenum(typeid(bool))); 
                char * ptr = PyArray_BYTES(ret);                            
                memcpy(ptr, &(val[0]), val.size() * sizeof(bool));            
                return PyArray_Return(ret);                                 
        } // ID_GETVEC

            case 'c': ID_GETVEC(char);
            case 'i': ID_GETVEC(int);
            case 'h': ID_GETVEC(short);
            case 'l': ID_GETVEC(long)
            case 'I': ID_GETVEC(unsigned int);
            case 'k': ID_GETVEC(unsigned long);
            case 'f': ID_GETVEC(float);
            case 'd': ID_GETVEC(double);
            case 's': {
                vector<string> val;
                Field<string>::getVec(self->id_, string(field), val);
                PyObject * ret = PyTuple_New(val.size());
                for (unsigned int ii = 0; ii < val.size(); ++ii){
                    PyTuple_SetItem(ret, (Py_ssize_t)ii, Py_BuildValue("s", val[ii].c_str()));
                }
                return ret;
            }
            default:
                PyErr_SetString(PyExc_ValueError, "Unhandled field type.");
                return NULL;                
        }
    }
    
    static PyObject * moose_Id_setField(_Id * self, PyObject * args)
    {
        PyObject * field = NULL;
        PyObject * value = NULL;
        if (!PyArg_ParseTuple(args, "OO:moose_Id_setField", &field, &value)){
            return NULL;
        }
        if (moose_Id_setattro(self, field, value) == -1){
            return NULL;
        }
        Py_RETURN_NONE;
    }

    static int moose_Id_setattro(_Id * self, PyObject * attr, PyObject *value)
    {
        char * fieldname = NULL;
        if (PyString_Check(attr)){
            fieldname = PyString_AsString(attr);
        } else {
            PyErr_SetString(PyExc_TypeError, "Attribute name must be a string");
            return -1;
        }
        string moose_class = Field<string>::get(self->id_, "class");
        string fieldtype = getFieldType(moose_class, string(fieldname), "valueFinfo");
        if (fieldtype.length() == 0){
            // If it is instance of a MOOSE Id then throw
            // error (to avoid silently creating new attributes due to
            // typos). Otherwise, it must have been subclassed in
            // Python. Then we allow normal Pythonic behaviour and
            // consider such mistakes user's responsibility.
            string class_name = ((PyTypeObject*)PyObject_Type((PyObject*)self))->tp_name;
            if (class_name != "Id"){
                Py_INCREF(attr);
                int ret = PyObject_GenericSetAttr((PyObject*)self, attr, value);
                Py_DECREF(attr);
                return ret;
            }
            ostringstream msg;
            msg << "'" << moose_class << "' class has no field '" << fieldname << "'" << endl;
            PyErr_SetString(PyExc_AttributeError, msg.str().c_str());
            return -1;
        }
        char ftype = shortType(fieldtype);
        int ret = 0;
        switch(ftype){
            case 'b': {
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<int> field, specified value must be a sequence." );
                }
                Py_ssize_t length = PySequence_Length(value);
                vector<bool> _value;
                for (unsigned int ii = 0; ii < length; ++ii){
                    PyObject * _v = PySequence_GetItem(value, ii);
                    bool v = (Py_True ==_v) || (PyInt_AsLong(_v) != 0);
                    _value.push_back(v);
                }
                ret = Field< bool >::setVec(self->id_, string(fieldname), _value);
                break;
            }
            case 'c': {
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<int> field, specified value must be a sequence." );
                }
                Py_ssize_t length = PySequence_Length(value);
                vector<char> _value;
                for (unsigned int ii = 0; ii < length; ++ii){
                    char * _v = PyString_AsString(value);
                    if (_v && _v[0]){
                        _value.push_back(_v[0]);
                    } else {
                        ostringstream err;
                        err << ii << "-th element is not a character or string.";
                        PyErr_SetString(PyExc_ValueError, err.str().c_str());
                        return -1;
                    }
                }
                ret = Field< char >::setVec(self->id_, string(fieldname), _value);
                break;
            }
            case 'i': {
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<int> field, specified value must be a sequence." );
                }
                Py_ssize_t length = PySequence_Length(value);
                vector<int> _value;
                for (unsigned int ii = 0; ii < length; ++ii){
                    int v = PyInt_AsLong(PySequence_GetItem(value, ii));
                    _value.push_back(v);
                }
                ret = Field< int >::setVec(self->id_, string(fieldname), _value);
                break;
            }
            case 'h': {
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<short> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<short> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        short v = PyInt_AsLong(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< short >::setVec(self->id_, string(fieldname), _value);
                }
                break;
            }
            case 'l': {//SET_VECFIELD(long, l)
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError,
                                    "For setting vector<long> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<long> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        long v = PyInt_AsLong(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field<long>::setVec(self->id_, string(fieldname), _value);
                }
                break;
            }
            case 'u': {//SET_VECFIELD(unsigned int, I)
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<unsigned int> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<unsigned int> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        unsigned int v = PyInt_AsUnsignedLongMask(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< unsigned int >::setVec(self->id_, string(fieldname), _value);
                }
                break;
            }
            case 'I': {//SET_VECFIELD(unsigned long, k)
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<unsigned long> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<unsigned long> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        unsigned long v = PyInt_AsUnsignedLongMask(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< unsigned long >::setVec(self->id_, string(fieldname), _value);
                }
                break;
            }
            case 'f': {//SET_VECFIELD(float, f)
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<float> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<float> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        float v = PyFloat_AsDouble(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field<float>::setVec(self->id_, string(fieldname), _value);
                }
                break;
            }
            case 'd': {//SET_VECFIELD(double, d)
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<double> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<double> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        double v = PyFloat_AsDouble(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field<double>::setVec(self->id_, string(fieldname), _value);
                }
                break;
            }                
            case 's': {
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<string> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<string> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        char * v = PyString_AsString(PySequence_GetItem(value, ii));
                        _value.push_back(string(v));
                    }
                    ret = Field<string>::setVec(self->id_, string(fieldname), _value);
                }
                break;
            }
            default:                
                break;
        }
        // MOOSE Field::set returns 1 for success 0 for
        // failure. Python treats return value 0 from stters as
        // success, anything else failure.
        if (ret){
            return 0;
        } else {
            ostringstream msg;
            msg <<  "Failed to set field '"  << fieldname << "'";
            PyErr_SetString(PyExc_AttributeError,msg.str().c_str());
            return -1;
        }
        
    }
    
    /////////////////////////////////////////////////////
    // ObjId functions.
    /////////////////////////////////////////////////////

    static int moose_ObjId_init_from_id(PyObject * self, PyObject * args, PyObject * kwargs)
    {
        extern PyTypeObject ObjIdType;
        // The char arrays are to avoid deprecation warning
        char _id[] = "id";
        char _dataIndex[] = "dataIndex";
        char _fieldIndex[] = "fieldIndex";
        char _numFieldBits[] = "numFieldBits";
        static char * kwlist[] = {
            _id, _dataIndex, _fieldIndex, _numFieldBits, NULL};
        _ObjId * instance = (_ObjId*)self;
        unsigned int id = 0, data = 0, field = 0, numFieldBits = 0;
        PyObject * obj = NULL;
        if (PyArg_ParseTupleAndKeywords(args, kwargs,
                                        "I|III:moose_ObjId_init",
                                        kwlist,
                                        &id, &data, &field, &numFieldBits)){
            instance->oid_ = ObjId(Id(id), DataId(data, field, numFieldBits));
            return 0;
        }
        PyErr_Clear();
        if (PyArg_ParseTupleAndKeywords(args, kwargs, "O|III:moose_ObjId_init",
                                        kwlist,
                                        &obj, &data, &field, &numFieldBits)){
            // If first argument is an Id object, construct an ObjId out of it
            if (Id_Check(obj)){
                instance->oid_ = ObjId(((_Id*)obj)->id_,
                                       DataId(data, field, numFieldBits));
                return 0;
            } else if (ObjId_SubtypeCheck(obj)){
                instance->oid_ = ((_ObjId*)obj)->oid_;
                return 0;
            }
        }
        return -1;
    }

    static int moose_ObjId_init_from_path(PyObject * self, PyObject * args,
                                          PyObject * kwargs)
    {
        extern PyTypeObject ObjIdType;        
        PyObject * dims = NULL;
        char * path = NULL;
        char * type = NULL;
        char _path[] = "path";
        char _dtype[] = "dtype";
        char _dims[] = "dims";
        static char * kwlist [] = {_path, _dims, _dtype, NULL};
        _ObjId * instance = (_ObjId*)self;
        instance->oid_ = ObjId::bad;
        if (kwargs == NULL){
            if (!PyArg_ParseTuple(args,
                                  "s|Os:moose_ObjId_init_from_path",
                                  &path,
                                  &dims,
                                  &type)){
                return -1;
            }
        } else if (!PyArg_ParseTupleAndKeywords(args,
                                                kwargs,
                                                "s|Os:moose_ObjId_init",
                                                kwlist,
                                                &path,
                                                &dims,
                                                &type)){
            return -1;
        }        
        instance->oid_ = ObjId(path);
        if (!(ObjId::bad == instance->oid_)){
            return 0;
        }
        // Path does not exist, create neww
        string basetype_str;
        if (type == NULL){
            // No type specified, go through the class hierarchy
            for (PyTypeObject * base = self->ob_type;
                 base != &ObjIdType; base = base->tp_base){
                basetype_str = base->tp_name;
                size_t dot = basetype_str.find('.');
                basetype_str = basetype_str.substr(dot+1);
                if (get_moose_classes().find(basetype_str) !=
                    get_moose_classes().end()){
                    break;
                }
            }
        } else {
            basetype_str = type;
        }
        if (dims == NULL){
            dims = Py_BuildValue("(i)", 1);
        }
        // Create an object using moose_Id_init method
        // and use the id to create a ref to first entry
        _Id * new_id = (_Id*)PyObject_New(_Id, &IdType);
        PyObject * newargs = PyTuple_New(3);
        if (newargs == NULL){
            return -1;
        }
        PyObject * pypath = PyString_FromString(path);
        if (pypath == NULL){
            return -1;
        }
        if (PyTuple_SetItem(newargs, 0, pypath)){
            return -1;
        }
        if (PyTuple_SetItem(newargs, 1, dims)){
            return -1;
        }
        PyObject * pybase = PyString_FromString(basetype_str.c_str());
        if (pybase == NULL){
            return -1;
        }
        if (PyTuple_SetItem(newargs, 2, pybase)){
            return -1;
        }        
        int ret = moose_Id_init(new_id, newargs, NULL);
        Py_DECREF(newargs);
        if (ret == 0){
            instance->oid_ = ObjId(new_id->id_);
            Py_DECREF(new_id);
        }
        return ret;            
    }
        
    PyDoc_STRVAR(moose_ObjId_init_documentation,
                 "__init__(path, dims, dtype) or"
                 " __init__(id, dataIndex, fieldIndex, numFieldBits)\n"
                 "Initialize moose object\n"
                 "Parameters\n"
                 "----------\n"
                 "path : string\n"
                 "Target element path.\n"
                 "dims : tuple or int\n"
                 "dimensions along each axis (can be"
                 " an integer for 1D objects). Default: (1,)\n"
                 "dtype : string\n"
                 "the MOOSE class name to be created.\n"
                 "id : Id or integer\n"
                 "id of an existing element.\n"
                 "\n");
        
    static int moose_ObjId_init(PyObject * self, PyObject * args,
                                PyObject * kwargs)
    {
        extern PyTypeObject ObjIdType;
        if (self && !ObjId_SubtypeCheck(self)){
            ostringstream error;
            error << "Expected an ObjId or subclass. Found "
                  << self->ob_type->tp_name;
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            return -1;
        }
        int ret = moose_ObjId_init_from_path(self, args, kwargs);
        if ( ret == 0){
            return 0;
        } else if (ret < -1){
            return -1;
        }
        PyErr_Clear();
        if (moose_ObjId_init_from_id(self, args, kwargs) == 0){
            return 0;
        }
        PyErr_SetString(PyExc_ValueError,
                        "Could not parse arguments. "
                        " Call __init__(path, dims, dtype) or"
                        " __init__(id, dataIndex, fieldIndex, numFieldBits)");        
        return -1;
    }

    /**
       This function simple returns the python hash of the unique path
       of this object.
    */
    static long moose_ObjId_hash(_ObjId * self)
    {
        PyObject * path = Py_BuildValue("s", self->oid_.path().c_str());        
        long ret = PyObject_Hash(path);
        Py_XDECREF(path);
        return ret;
    }
    
    static PyObject * moose_ObjId_repr(_ObjId * self)
    {
#ifdef HAVE_LONG_LONG
        return PyString_FromFormat("<ObjId: id=%u, dataId=%llu, path=%s>",
                                   self->oid_.id.value(),
                                   self->oid_.dataId.value(),
                                   self->oid_.path().c_str());
#else
        return PyString_FromFormat("<ObjId: id=%u, dataId=%lu, path=%s>",
                                   self->oid_.id.value(),
                                   self->oid_.dataId.value(),
                                   self->oid_.path().c_str());
#endif
    } // !  moose_ObjId_repr


    PyDoc_STRVAR(moose_ObjId_getId_documentation,
                 "getId()\n"
                 "\n"
                 "Get the Id of this object\n"
                 "\n");
    static PyObject* moose_ObjId_getId(_ObjId * self)
    {
        extern PyTypeObject IdType;        
        _Id * ret = PyObject_New(_Id, &IdType);
        ret->id_ = self->oid_.id;
        return (PyObject*)ret;
    }

    PyDoc_STRVAR(moose_ObjId_getFieldType_documentation,
                 "getFieldType(fieldName, finfoType='valueFinfo')\n"
                 "\n"
                 "Get the string representation of the type of this field.\n"
                 "\n"
                 "Parameters\n"
                 "----------"
                 "fieldName : string\n"
                 "\tName of the field to be queried.\n"
                 "finfoType : string\n"
                 "\tFinfotype the field should be looked in for (can be \n"
                 "valueFinfo, srcFinfo, destFinfo, lookupFinfo)\n"
                 "\n");
    
    static PyObject * moose_ObjId_getFieldType(_ObjId * self, PyObject * args)
    {
        char * fieldName = NULL;
        char * finfoType = NULL;
        if (!PyArg_ParseTuple(args, "s|s:moose_ObjId_getFieldType", &fieldName,
                              &finfoType)){
            return NULL;
        }
        string finfoTypeStr = "";
        if (finfoType != NULL){
            finfoTypeStr = finfoType;
        } else {
            finfoTypeStr = "valueFinfo";
        }
        string typeStr = getFieldType(Field<string>::get(self->oid_, "class"),
                                      string(fieldName), finfoTypeStr);
        if (typeStr.length() <= 0){
            PyErr_SetString(PyExc_ValueError,
                            "Empty string for field type. "
                            "Field name may be incorrect.");
            return NULL;
        }
        PyObject * type = PyString_FromString(typeStr.c_str());
        return type;
    }  // ! moose_Id_getFieldType

    /**
       Wrapper over getattro to allow direct access as a function with variable argument list
     */
    
    PyDoc_STRVAR(moose_ObjId_getField_documentation,
                 "getField(fieldName)\n"
                 "\n"
                 "Get the value of the field.\n"
                 "\n"
                 "Parameters\n"
                 "----------"
                 "fieldName : string\n"
                 "\tName of the field.");
    static PyObject * moose_ObjId_getField(_ObjId * self, PyObject * args)
    {
        PyObject * attr;        
        if (!PyArg_ParseTuple(args, "O:moose_ObjId_getField", &attr)){
            return NULL;
        }
        return moose_ObjId_getattro(self, attr);
    }

    /**
       2011-03-28 13:59:41 (+0530)
       
       Get a specified field. Re-done on: 2011-03-23 14:42:03 (+0530)

       I wonder how to cleanly do this. The Id - ObjId dichotomy is
       really ugly. When you don't pass an index, it is just treated
       as 0. Then what is the point of having Id separately? ObjId
       would been just fine!
    */
    static PyObject * moose_ObjId_getattro(_ObjId * self, PyObject * attr)
    {
        extern PyTypeObject IdType;
        extern PyTypeObject ObjIdType;
        const char * field;
        char ftype;
        if(!ObjId_SubtypeCheck(self)){
            cerr << "Expected an ObjId or subclass. But found `" << ((PyObject*)self)->ob_type->tp_name << endl;
            return NULL;
        }
        if (PyString_Check(attr)){
            field = PyString_AsString(attr);
        } else {
            return PyObject_GenericGetAttr((PyObject*)self, attr);
        }
        PyObject * _ret = get_ObjId_attr(self, field);
        if (_ret != NULL){
            return _ret;
        }
        // The GET_FIELD macro is just a short-cut to reduce typing
        // TYPE is the full type string for the field. TYPEC is the corresponding Python Py_BuildValue format character.
#define GET_FIELD(TYPE, TYPEC)                                          \
        {                                                               \
            TYPE value = Field<TYPE>::get(self->oid_, string(field));   \
            PyObject * ret = Py_BuildValue(#TYPEC, value);              \
            return ret;                                                 \
        }                                                               \


#define GET_VECFIELD(TYPE)                                              \
        {                                                               \
                vector<TYPE> val = Field< vector<TYPE> >::get(self->oid_, string(field)); \
                npy_intp dims = val.size();                             \
                PyArrayObject * ret = (PyArrayObject*)PyArray_SimpleNew(1, &dims, get_npy_typenum(typeid(TYPE))); \
                char * ptr = PyArray_BYTES(ret);        \
                memcpy(ptr, &(val[0]), val.size() * sizeof(TYPE));      \
                return PyArray_Return(ret);                             \
        }                                                               \

        
#define GET_VECVEC(TYPE, TYPEC){                                        \
                vector < vector <TYPE> > val = Field< vector < vector < TYPE > > >::get(self->oid_, string(field)); \
                Py_ssize_t length1 = val.size();                        \
                PyObject * ret = PyList_New(length1);                   \
                ostringstream error;                                    \
                error << "moose_ObjId_getattro: ";                      \
                for (Py_ssize_t ii = 0; ii < length1; ++ii){            \
                    Py_ssize_t length2 = val[ii].size();            \
                    PyObject * subseq = PyList_New(length2);        \
                    if ( PyList_SetItem(ret, ii, subseq) != 0 ){    \
                        Py_DECREF(ret);                         \
                        error << "Failed to set subsequence " << ii; \
                        PyErr_SetString(PyExc_RuntimeError, error.str().c_str());\
                    }                                                                   \
                    for (Py_ssize_t jj = 0; jj < length2; ++jj){                        \
                        PyObject * value = Py_BuildValue(#TYPEC, val[ii][jj]);          \
                        if (value == NULL){                                             \
                            Py_DECREF(ret);                                             \
                            error << "failed to convert "                               \
                                  << field << "[" << ii << "][" << jj << "]";           \
                            PyErr_SetString(PyExc_RuntimeError, error.str().c_str());   \
                            return NULL;                                                \
                        }                                                               \
                        if (PyList_SetItem(subseq, jj, value) != 0){                    \
                            Py_DECREF(ret);                                             \
                            error << "could not set item [" << ii << "][" << jj << "]"; \
                            PyErr_SetString(PyExc_RuntimeError, error.str().c_str());   \
                            return NULL;                                                \
                        }                                                               \
                    }                                                                   \
                }                                                                       \
            return ret;                                                             \
        } // GET_VECVEC
        assert(!(self->oid_ == ObjId::bad));
        string class_name = Field<string>::get(self->oid_, "class");
        string type = getFieldType(class_name, string(field), "valueFinfo");
        if (type.empty()){
            // Check if this field name is aliased and update fieldname and type if so.
            map<string, string>::const_iterator it = get_field_alias().find(string(field));
            if (it != get_field_alias().end()){
                field = (it->second).c_str();
                type = getFieldType(Field<string>::get(self->oid_, "class"), it->second, "valueFinfo");
                // Update attr for next level (PyObject_GenericGetAttr) in case.
                Py_XDECREF(attr);
                attr = PyString_FromString(field);
            }
        }
        if (type.empty()){
            return PyObject_GenericGetAttr((PyObject*)self, attr);            
        }
        ftype = shortType(type);
        if (!ftype){
            return PyObject_GenericGetAttr((PyObject*)self, attr);
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
                return Py_BuildValue("s", _s.c_str());
            }
            case 'x':
                {                    
                    Id value = Field<Id>::get(self->oid_, string(field));
                    _Id * ret = PyObject_New(_Id, &IdType);
                    ret->id_ = value;
                    return (PyObject*)ret;
                }
            case 'y':
                {
                    ObjId value = Field<ObjId>::get(self->oid_, string(field));
                    _ObjId * ret = PyObject_New(_ObjId, &ObjIdType);
                    ret->oid_ = value;
                    return (PyObject*)ret;
                }
            case 'z':
                {
                    PyErr_SetString(PyExc_NotImplementedError, "DataId handling not implemented yet.");
                    return NULL;
                }
            case 'D': GET_VECFIELD(double)
            case 'F': GET_VECFIELD(float)
            case 'v': GET_VECFIELD(int)
            case 'L': GET_VECFIELD(long)
            case 'K': GET_VECFIELD(unsigned long)
            case 'U': GET_VECFIELD(unsigned int)
            case 'w': GET_VECFIELD(short)
            case 'C': GET_VECFIELD(char)
            case 'S': {                                                 
                vector<string> val = Field< vector<string> >::get(self->oid_, string(field)); 
                PyObject * ret = PyTuple_New((Py_ssize_t)val.size());
                
                for (unsigned int ii = 0; ii < val.size(); ++ ii ){     
                    PyObject * entry = Py_BuildValue("s", val[ii].c_str()); 
                    if (!entry || PyTuple_SetItem(ret, (Py_ssize_t)ii, entry)){
                        Py_XDECREF(entry);
                        Py_XDECREF(ret);                                  
                        return NULL;                                
                    }                                               
                }                                                       
                return ret;                                                  
            }
            case 'X': // vector<Id>
                {
                    vector<Id> value = Field< vector <Id> >::get(self->oid_, string(field));
                    PyObject * ret = PyTuple_New((Py_ssize_t)value.size());
                
                    for (unsigned int ii = 0; ii < value.size(); ++ii){
                        _Id * entry = PyObject_New(_Id, &IdType);
                        if (!entry){
                            Py_XDECREF(ret);
                            return NULL;
                        }
                        entry->id_ = value[ii];
                        if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){
                            Py_XDECREF(entry);
                            Py_XDECREF(ret);
                            return NULL;
                        }
                    }
                    return ret;
                }
            case 'Y': // vector<ObjId>
                {
                    vector<ObjId> value = Field< vector <ObjId> >::get(self->oid_, string(field));
                    PyObject * ret = PyTuple_New(value.size());
                
                    for (unsigned int ii = 0; ii < value.size(); ++ii){
                        _ObjId * entry = PyObject_New(_ObjId, &ObjIdType);                       
                        if (!entry){
                            Py_XDECREF(ret);
                            return NULL;
                        }
                        entry->oid_ = value[ii];
                        if (PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){
                            Py_XDECREF(entry);
                            Py_XDECREF(ret);
                            return NULL;
                        }
                    }
                    return ret;
                }
            case 'P': // vector < vector < unsigned > >
                {
                    GET_VECVEC(unsigned int, I);
                    break;
                }
            case 'Q': // vector < vector < int > >
                {
                    GET_VECVEC(int, i);                
                    break;
                }
            case 'R': // vector < vector <double> >
                {
                    GET_VECVEC(double, d);
                    break;
                }
            default:
                    return PyObject_GenericGetAttr((PyObject*)self, attr);
        }
#undef GET_FIELD    
#undef GET_VECFIELD
        return NULL;        
    }

    /**
       Wrapper over setattro to make METHOD_VARARG
    */
    PyDoc_STRVAR(moose_ObjId_setField_documentation,
                 "setField(fieldName, value)\n"
                 "\n"
                 "Set the value of specified field.\n"
                 "\n"
                 "Parameters\n"
                 "----------"
                 "fieldName : string\n"
                 "\tField to be assigned value to.\n"
                 "value : python datatype compatible with the type of the field\n"
                 "\tThe value to be assigned to the field.");
    
    static PyObject * moose_ObjId_setField(_ObjId * self, PyObject * args)
    {
        PyObject * field;
        PyObject * value;
        if (!PyArg_ParseTuple(args, "OO:moose_ObjId_setField", &field, &value)){
            return NULL;
        }
        if (moose_ObjId_setattro(self, field, value) == -1){
            return NULL;
        }
        Py_RETURN_NONE;
    }
    
    /**
       Set a specified field. Redone on 2011-03-23 14:41:45 (+0530)
    */
    static int  moose_ObjId_setattro(_ObjId * self, PyObject * attr, PyObject * value)
    {
        const char * field;
        if (PyString_Check(attr)){
            field = PyString_AsString(attr);
        } else {
            PyErr_SetString(PyExc_TypeError, "Attribute name must be a string");
            return -1;
        }
        string fieldtype = getFieldType(Field<string>::get(self->oid_, "class"), string(field), "valueFinfo");
        if (fieldtype.length() == 0){
            // If it is instance of a MOOSE built-in class then throw
            // error (to avoid silently creating new attributes due to
            // typos). Otherwise, it must have been subclassed in
            // Python. Then we allow normal Pythonic behaviour and
            // consider such mistakes user's responsibility.
            string class_name = ((PyTypeObject*)PyObject_Type((PyObject*)self))->tp_name;            
            if (get_moose_classes().find(class_name) == get_moose_classes().end()){
                return PyObject_GenericSetAttr((PyObject*)self, PyString_FromString(field), value);
            }
            ostringstream msg;
            msg << "'" << class_name << "' class has no field '" << field << "'" << endl;
            PyErr_SetString(PyExc_AttributeError, msg.str().c_str());
            return -1;
        }
        char ftype = shortType(fieldtype);
        int ret = 0;
        switch(ftype){
            case 'b': {
                bool _value = (Py_True == value) || (PyInt_AsLong(value) != 0);
                ret = Field<bool>::set(self->oid_, string(field), _value);
                break;
            }
            case 'c': {
                char * _value = PyString_AsString(value);
                if (_value && _value[0]){
                    ret = Field<char>::set(self->oid_, string(field), _value[0]);
                }
                break;
            }
            case 'i': {
                int _value = PyInt_AsLong(value);
                if ((_value != -1) || (!PyErr_Occurred())){
                    ret = Field<int>::set(self->oid_, string(field), _value);
                }
                break;
            }
            case 'h':{
                short _value = (short)PyInt_AsLong(value);
                if ((_value != -1) || (!PyErr_Occurred())){
                    ret = Field<short>::set(self->oid_, string(field), _value);
                }
                break;
            }
            case 'l': {
                long _value = PyInt_AsLong(value);
                if ((_value != -1) || (!PyErr_Occurred())){
                    ret = Field<long>::set(self->oid_, string(field), _value);
                }
                break;
            }
            case 'I': {
                unsigned long _value = PyInt_AsUnsignedLongMask(value);
                ret = Field<unsigned int>::set(self->oid_, string(field), (unsigned int)_value);
                break;
            }
            case 'k': {
                unsigned long _value = PyInt_AsUnsignedLongMask(value);
                ret = Field<unsigned long>::set(self->oid_, string(field), _value);
                break;
            }                
            case 'f': {
                float _value = PyFloat_AsDouble(value);
                ret = Field<float>::set(self->oid_, string(field), _value);
                break;
            }
            case 'd': {
                double _value = PyFloat_AsDouble(value);
                ret = Field<double>::set(self->oid_, string(field), _value);
                break;
            }
            case 's': {
                char * _value = PyString_AsString(value);
                if (_value){
                    ret = Field<string>::set(self->oid_, string(field), string(_value));
                }
                break;
            }
            case 'x': {// Id
                if (value){
                    ret = Field<Id>::set(self->oid_, string(field), ((_Id*)value)->id_);
                } else {
                    PyErr_SetString(PyExc_ValueError, "Null pointer passed as Id value.");
                    return -1;
                }
                break;
            }
            case 'y': {// ObjId
                if (value){
                    ret = Field<ObjId>::set(self->oid_, string(field), ((_ObjId*)value)->oid_);
                } else {
                    PyErr_SetString(PyExc_ValueError, "Null pointer passed as Id value.");
                    return -1;
                }
                break;
            }
            case 'z': {// DataId
                PyErr_SetString(PyExc_NotImplementedError, "DataId handling not implemented yet.");
                return -1;
            }
            case 'v': {
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<int> field, specified value must be a sequence." );
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
            case 'w': {
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<short> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<short> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        short v = PyInt_AsLong(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < short > >::set(self->oid_, string(field), _value);
                }
                break;
            }
            case 'L': {//SET_VECFIELD(long, l)
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError,
                                    "For setting vector<long> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<long> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        long v = PyInt_AsLong(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < long > >::set(self->oid_, string(field), _value);
                }
                break;
            }
            case 'U': {//SET_VECFIELD(unsigned int, I)
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<unsigned int> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<unsigned int> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        unsigned int v = PyInt_AsUnsignedLongMask(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < unsigned int > >::set(self->oid_, string(field), _value);
                }
                break;
            }
            case 'K': {//SET_VECFIELD(unsigned long, k)
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<unsigned long> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<unsigned long> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        unsigned long v = PyInt_AsUnsignedLongMask(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < unsigned long > >::set(self->oid_, string(field), _value);
                }
                break;
            }
            case 'F': {//SET_VECFIELD(float, f)
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<float> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<float> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        float v = PyFloat_AsDouble(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < float > >::set(self->oid_, string(field), _value);
                }
                break;
            }
            case 'D': {//SET_VECFIELD(double, d)
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<double> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<double> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        double v = PyFloat_AsDouble(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                    ret = Field< vector < double > >::set(self->oid_, string(field), _value);
                }
                break;
            }                
            case 'S': {
                if (!PySequence_Check(value)){
                    PyErr_SetString(PyExc_TypeError, "For setting vector<string> field, specified value must be a sequence." );
                } else {
                    Py_ssize_t length = PySequence_Length(value);
                    vector<string> _value;
                    for (unsigned int ii = 0; ii < length; ++ii){
                        char * v = PyString_AsString(PySequence_GetItem(value, ii));
                        _value.push_back(string(v));
                    }
                    ret = Field< vector < string > >::set(self->oid_, string(field), _value);
                }
                break;
            }
            case 'P': {// vector< vector<unsigned int> >
                vector < vector <unsigned> > * _value = (vector < vector <unsigned> > *)to_cpp< vector < vector <unsigned> > >(value);
                if (!PyErr_Occurred()){
                    ret = Field < vector < vector <unsigned> > >::set(self->oid_, string(field), *_value);
                }
                delete _value;
                break;
            }
            case 'Q': {// vector< vector<int> >
                vector < vector <int> > * _value = (vector < vector <int> > *)to_cpp< vector < vector <int> > >(value);
                if (!PyErr_Occurred()){
                    ret = Field < vector < vector <int> > >::set(self->oid_, string(field), *_value);
                }
                delete _value;
                break;
            }
            case 'R': {// vector< vector<double> >
                vector < vector <double> > * _value = (vector < vector <double> > *)to_cpp< vector < vector <double> > >(value);
                if (!PyErr_Occurred()){
                    ret = Field < vector < vector <double> > >::set(self->oid_, string(field), *_value);
                }
                delete _value;
                break;
            }
            default:                
                break;
        }
        // MOOSE Field::set returns 1 for success 0 for
        // failure. Python treats return value 0 from stters as
        // success, anything else failure.
        if (ret){
            return 0;
        } else {
            ostringstream msg;
            msg <<  "Failed to set field '"  << field << "'";
            PyErr_SetString(PyExc_AttributeError,msg.str().c_str());
            return -1;
        }
    } // moose_ObjId_setattro

    /// Inner function for looking up value from LookupField on object
    /// with ObjId target.
    ///
    /// args should be a tuple (lookupFieldName, key)
    PyObject * getLookupField(ObjId target, char * fieldName, PyObject * key)
    {
        extern PyTypeObject ObjIdType;
        vector<string> type_vec;
        if (parse_Finfo_type(Field<string>::get(target, "class"), "lookupFinfo", string(fieldName), type_vec) < 0){
            ostringstream error;
            error << "Cannot handle key type for LookupField `" << Field<string>::get(target, "class") << "." << fieldName << "`.";
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            return NULL;
        }
        if (type_vec.size() != 2){
            ostringstream error;
            error << "LookupField type signature should be <keytype>, <valuetype>. But for `"
                  << Field<string>::get(target, "class") << "." << fieldName << "` got " << type_vec.size() << " components." ;
            PyErr_SetString(PyExc_AssertionError, error.str().c_str());
            return NULL;
        }
        PyObject * ret = NULL;
        char key_type_code = shortType(type_vec[0]);
        char value_type_code = shortType(type_vec[1]);
        void * value_ptr = NULL;
        if (value_type_code == 'x'){
            value_ptr = PyObject_New(_Id, &IdType);
        } else if (value_type_code == 'y'){
            value_ptr = PyObject_New(_ObjId, &ObjIdType);
        }
        switch(key_type_code){
            case 'b': {
                ret = lookup_value <bool> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'c': {
                ret = lookup_value <char> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'h': {
                ret = lookup_value <short> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }            
            case 'H': {
                ret = lookup_value <unsigned short> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }            
            case 'i': {
                ret = lookup_value <int> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }            
            case 'I': {
                ret = lookup_value <unsigned int> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }            
            case 'l': {
                ret = lookup_value <long> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }                        
            case 'k': {
                ret = lookup_value <unsigned long> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }                        
            case 'L': {
                ret = lookup_value <long long> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }                        
            case 'K': {
                ret = lookup_value <unsigned long long> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }                        
            case 'd': {
                ret = lookup_value <double> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }                        
            case 'f': {
                ret = lookup_value <float> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 's': {
                ret = lookup_value <string> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'x': {
                ret = lookup_value <Id> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'y': {
                ret = lookup_value <ObjId> (target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'v': {
                ret = lookup_value < vector <int> >(target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'w': {
                ret = lookup_value < vector <short> >(target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }   
            case 'U': {
                ret = lookup_value < vector <unsigned int> >(target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }
            case 'F': {
                ret = lookup_value < vector <float> >(target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }                
            case 'D': {
                ret = lookup_value < vector <double> >(target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }                
            case 'S': {
                ret = lookup_value < vector <string> >(target, string(fieldName), value_type_code, key_type_code, key, value_ptr);
                break;
            }                
            default:
                ostringstream error;
                error << "Unhandled key type `" << type_vec[0] << "` for " << Field<string>::get(target, "class") << "." << fieldName;
                PyErr_SetString(PyExc_TypeError, error.str().c_str());
        }                
        return ret;
    }

    PyDoc_STRVAR(moose_ObjId_getLookupField_documentation,
                 "getLookupField(fieldName, key)\n"
                 "\n"
                 "Lookup entry for `key` in `fieldName` \n"
                 "\n"
                 "Parameters\n"
                 "----------"
                 "fieldName : string\n"
                 "\tName of the lookupfield.\n"
                 "key : appropriate type for key of the lookupfield (as in the dict"
                 " getFieldDict).\n"
                 "\tKey for the look-up.");

    static PyObject * moose_ObjId_getLookupField(_ObjId * self, PyObject * args)
    {
        char * fieldName = NULL;
        PyObject * key = NULL;
        if (!PyArg_ParseTuple(args, "sO:moose_ObjId_getLookupField", &fieldName,  &key)){
            return NULL;
        }
        return getLookupField(self->oid_, fieldName, key);
    } // moose_ObjId_getLookupField

    int setLookupField(ObjId target, char * fieldName, PyObject * key, PyObject * value)
    {
        vector<string> type_vec;
        if (parse_Finfo_type(Field<string>::get(target, "class"), "lookupFinfo", string(fieldName), type_vec) < 0){
            ostringstream error;
            error << "Cannot handle key type for LookupField `" << Field<string>::get(target, "class") << "." << fieldName << "`.";
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            return -1;
        }
        if (type_vec.size() != 2){
            ostringstream error;
            error << "LookupField type signature should be <keytype>, <valuetype>. But for `"
                  << Field<string>::get(target, "class") << "." << fieldName << "` got " << type_vec.size() << " components." ;
            PyErr_SetString(PyExc_AssertionError, error.str().c_str());
            return -1;
        }
        char key_type_code = shortType(type_vec[0]);
        char value_type_code = shortType(type_vec[1]);
        int ret = -1;
        switch(key_type_code){
            case 'b': {
                ret = set_lookup_value <bool> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }
            case 'c': {
                ret = set_lookup_value <char> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }
            case 'h': {
                ret = set_lookup_value <short> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }            
            case 'H': {
                ret = set_lookup_value <unsigned short> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }            
            case 'i': {
                ret = set_lookup_value <int> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }            
            case 'I': {
                ret = set_lookup_value <unsigned int> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }            
            case 'l': {
                ret = set_lookup_value <long> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 'k': {
                ret = set_lookup_value <unsigned long> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 'L': {
                ret = set_lookup_value <long long> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 'K': {
                ret = set_lookup_value <unsigned long long> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 'd': {
                ret = set_lookup_value <double> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 'f': {
                ret = set_lookup_value <float> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }
            case 'x': {
                ret = set_lookup_value <Id> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }
            case 'y': {
                ret = set_lookup_value <ObjId> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }
            default:
                ostringstream error;
                error << "setLookupField: invalid key type " << type_vec[0];
                PyErr_SetString(PyExc_TypeError, error.str().c_str());
        }
        return ret;        
    }// setLookupField

    PyDoc_STRVAR(moose_ObjId_setLookupField_documentation,
                 "setLookupField(field, key, value)\n"
                 "Set a lookup field entry.\n"
                 "Parameters\n"
                 "----------\n"
                 "field : string\n"
                 "\tname of the field to be set\n"
                 "key : key type\n"
                 "\tkey in the lookup field for which the value is to be set.\n"
                 "value : value type\n"
                 "\tvalue to be set for `key` in the lookkup field.");
    
    static PyObject * moose_ObjId_setLookupField(_ObjId * self, PyObject * args)
    {
        PyObject * key;
        PyObject * value;
        char * field;
        if (!PyArg_ParseTuple(args, "sOO:moose_ObjId_setLookupField", &field,  &key, &value)){
            return NULL;
        }
        if ( setLookupField(self->oid_, field, key, value) == 0){
            Py_RETURN_NONE;
        }
        return NULL;
    }// moose_ObjId_setLookupField

    PyDoc_STRVAR(moose_ObjId_setDestField_documentation,
                 "setDestField(arg0, arg1, ...)\n"
                 "Set a destination field. This is for advanced uses. destFields can\n"
                 "(and should) be directly called like functions as\n"
                 "`ObjId.fieldname(arg0, ...)`\n"
                 "\n"
                 "Parameters\n"
                 "----------\n"
                 "The number and type of paramateres depend on the destFinfo to be\n"
                 "set. Use moose.doc('{classname}.{fieldname}') to get builtin\n"
                 "documentation on the destFinfo `fieldname`\n"
                 );
    
    static PyObject * moose_ObjId_setDestField(PyObject * self, PyObject * args)
    {           
        return _setDestField(((_ObjId*)self)->oid_, args);        
    }

    
    static PyObject * _setDestField(ObjId oid, PyObject *args)
    {
        PyObject * arglist[10] = {NULL, NULL, NULL, NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL};
        ostringstream error;
        error << "moose.setDestField: ";
        // Unpack the arguments
        if (!PyArg_UnpackTuple(args, "setDestField", minArgs, maxArgs,
                               &arglist[0], &arglist[1], &arglist[2],
                               &arglist[3], &arglist[4], &arglist[5],
                               &arglist[6], &arglist[7], &arglist[8],
                               &arglist[9])){
            error << "At most " << maxArgs - 1 << " arguments can be handled.";
            PyErr_SetString(PyExc_ValueError, error.str().c_str());
            return NULL;
        }
        
        // Get the destFinfo name
        char * fieldName = PyString_AsString(arglist[0]);
        if (!fieldName){ // not a string, raises TypeError
            error << "first argument must be a string specifying field name.";
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            return NULL;
        }
        
        // Try to parse the arguments.
        vector< string > argType;
        if (parse_Finfo_type(Field<string>::get(oid, "class"),
                             "destFinfo", string(fieldName), argType) < 0){
            error << "Arguments not handled: " << fieldName << "(";
            for (unsigned int ii = 0; ii < argType.size(); ++ii){
                error << argType[ii] << ",";
            }
            error << ")";
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            return NULL;
        }
        // Construct the argument list
        ostringstream argstream;
        for (size_t ii = 0; ii < argType.size(); ++ii){
            PyObject * arg = arglist[ii+1];
            if ( arg == NULL && argType[ii] == "void"){
                bool ret = SetGet0::set(oid, string(fieldName));
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
                    ////////////////////////////////////////////////////
                    // We do handle multiple vectors. Use the argument
                    // list as a single vector argument.
                    ////////////////////////////////////////////////////
                case 'v': 
                    {
                        return _set_vector_destFinfo<int>(oid, string(fieldName), ii, arg);
                        break;
                    }
                case 'w': 
                    {
                        return _set_vector_destFinfo<short>(oid, string(fieldName), ii, arg);
                        break;
                    }
                case 'L': //SET_VECFIELD(long, l)
                    {
                        return _set_vector_destFinfo<long>(oid, string(fieldName), ii, arg);
                        break;
                    }
                case 'U'://SET_VECFIELD(unsigned int, I)
                    {
                        return _set_vector_destFinfo<unsigned int>(oid, string(fieldName), ii, arg);
                        break;
                    }
                case 'K': //SET_VECFIELD(unsigned long, k)
                    {
                        return _set_vector_destFinfo<unsigned long>(oid, string(fieldName), ii, arg);
                        break;
                    }
                case 'F': //SET_VECFIELD(float, f)
                    {
                        return _set_vector_destFinfo<float>(oid, string(fieldName), ii, arg);
                        break;
                    }
                case 'D': //SET_VECFIELD(double, d)
                    {
                        return _set_vector_destFinfo<double>(oid, string(fieldName), ii, arg);
                        break;
                    }                
                case 'S':
                    {
                        return _set_vector_destFinfo<string>(oid, string(fieldName), ii, arg);
                        break;
                    }
                case 'X':
                    {
                        return _set_vector_destFinfo<Id>(oid, string(fieldName), ii, arg);
                        break;
                    }
                case 'Y':
                    {
                        return _set_vector_destFinfo<ObjId>(oid, string(fieldName), ii, arg);
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
        bool ret = SetGet::strSet(oid, string(fieldName), argstring);
        if (ret){
            Py_RETURN_TRUE;
        } else {
            Py_RETURN_FALSE;
        }        
    } // moose_ObjId_setDestField

    PyDoc_STRVAR(moose_ObjId_getFieldNames_documenation,
                 "getFieldNames(fieldType='')\n"
                 "\n"
                 "Get the names of fields on this ObjId.\n"
                 "\n"
                 "Parameters\n"
                 "----------\n"
                 "fieldType : str\n"
                 "\tField type to retrieve. Can be `valueFinfo`, `srcFinfo`,\n"
                 "\t`destFinfo`, `lookupFinfo`, etc. If an empty string is specified,\n"
                 "\tnames of all avaialable fields are returned.\n"
                 "\n"
                 "Returns\n"
                 "-------\n"
                 "\tout : tuple of strings.\n"
                 "\n"
                 "Example\n"
                 "-------\n"
                 "List names of all the source fields in PulseGen class:\n"
                 ">>> moose.getFieldNames('PulseGen', 'srcFinfo')\n"
                 "('childMsg', 'outputOut')\n"
                 "\n");
    // 2011-03-23 15:28:26 (+0530)
    static PyObject * moose_ObjId_getFieldNames(_ObjId * self, PyObject *args)
    {
        char * ftype = NULL;
        if (!PyArg_ParseTuple(args, "|s:moose_ObjId_getFieldNames", &ftype)){
            return NULL;
        }
        string ftype_str = (ftype != NULL)? string(ftype): "";
        vector<string> ret;
        string className = Field<string>::get(self->oid_, "class");
        if (ftype_str == ""){
            for (const char **a = getFinfoTypes(); *a; ++a){
                vector<string> fields = getFieldNames(className, string(*a));
                ret.insert(ret.end(), fields.begin(), fields.end());
            }            
        } else {
            ret = getFieldNames(className, ftype_str);
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
                Py_XDECREF(fname);
                Py_XDECREF(pyret);
                pyret = NULL;
                break;
            }
        }
        return pyret;             
    }

    PyDoc_STRVAR(moose_ObjId_getNeighbors_documentation,
                 "getNeighbours(fieldName)\n"
                 "\n"
                 "Get the objects connected to this ObjId by a message on specified\n"
                 "field.\n"
                 "\n"
                 "Parameters\n"
                 "----------\n"
                 "fieldName : str\n"
                 "\tname of the connection field (a destFinfo or srcFinfo)\n"
                 "\n"
                 "Returns\n"
                 "-------\n"
                 "out: tuple of Id-s.\n"
                 "\n");
                 
    static PyObject * moose_ObjId_getNeighbors(_ObjId * self, PyObject * args)
    {
        char * field = NULL;
        if (!PyArg_ParseTuple(args, "s:moose_ObjId_getNeighbors", &field)){
            return NULL;
        }
        vector< Id > val = LookupField< string, vector< Id > >::get(self->oid_, "neighbours", string(field));
    
        PyObject * ret = PyTuple_New((Py_ssize_t)val.size());
                
        for (unsigned int ii = 0; ii < val.size(); ++ ii ){            
            _Id * entry = PyObject_New(_Id, &IdType);
            if (!entry || PyTuple_SetItem(ret, (Py_ssize_t)ii, (PyObject*)entry)){
                Py_XDECREF(entry);
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
    PyDoc_STRVAR(moose_ObjId_connect_documentation,
                 "connect(srcfield, destobj, destfield, msgtype) -> bool\n"
                 "Connect another object via a message.\n"
                 "Parameters\n"
                 "----------\n"
                 "srcfield : str\n"
                 "\tsource field on self.\n"
                 "destobj : ObjId\n"
                 "\tDestination object to connect to.\n"
                 "destfield : str\n"
                 "\tfield to connect to on `destobj`.\n"
                 "msgtype : str\n"
                 "\ttype of the message. Can be `Single`, `OneToAll`, `AllToOne`,\n"
                 " `OneToOne`, `Reduce`, `Sparse`. Default: `Single`."
                 "\n"
                 "Returns\n"
                 "-------\n"
                 "True on success, False on failure\n"
                 "\n"
                 "See also\n"
                 "--------\n"
                 "moose.connect\n"
                 "\n"
                 );
    static PyObject * moose_ObjId_connect(_ObjId * self, PyObject * args)
    {
        PyObject * destPtr = NULL;
        char * srcField = NULL, * destField = NULL, * msgType = NULL;
        static char default_msg_type[] = "Single";
        if(!PyArg_ParseTuple(args,
                             "sOs|s:moose_ObjId_connect",
                             &srcField,
                             &destPtr,
                             &destField,
                             &msgType)){
            return NULL;
        }
        if (msgType == NULL){
            msgType = default_msg_type;
        }
        _ObjId * dest = reinterpret_cast<_ObjId*>(destPtr);
        bool ret = (ShellPtr->doAddMsg(msgType,
                                       self->oid_,
                                       string(srcField),
                                       dest->oid_,
                                       string(destField)) !=
                    Msg::bad);
        if (!ret){
            PyErr_SetString(PyExc_NameError,
                            "connect failed: check field names and type compatibility.");
            return NULL;
        }
        return Py_BuildValue("i", ret);
    }

    PyDoc_STRVAR(moose_ObjId_compare_documentation,
                 "Compare two ObjId instances. This just does a string comparison of\n"
                 "the paths of the ObjId instances. This function exists only to\n"
                 "facilitate certain operations requiring sorting/comparison, like\n"
                 "using ObjIds for dict keys. Conceptually only equality comparison is\n"
                 "meaningful for ObjIds.\n"); 
    static int moose_ObjId_compare(_ObjId * self, PyObject * other)
    {
        extern PyTypeObject ObjIdType;
        int ret;
        if (self != NULL && other == NULL){
            return 1;
        } else if (self == NULL && other != NULL){
            return -1;
        } else if (!ObjId_SubtypeCheck(other)){
            ostringstream error;
            error << "Cannot compare ObjId with "
                  << other->ob_type->tp_name;
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            ret = -1;
        }
        string l_path = self->oid_.path();
        string r_path = ((_ObjId*)other)->oid_.path();
        return l_path.compare(r_path);
    }

    PyDoc_STRVAR(moose_ObjId_getDataIndex_documentation,
                 "getDataIndex()\n"
                 "\n"
                 "Return the dataIndex of this object.\n");
    static PyObject * moose_ObjId_getDataIndex(_ObjId * self)
    {
        PyObject * ret = Py_BuildValue("I", self->oid_.dataId.value());
        return ret;
    }

    // WARNING: fieldIndex has been deprecated in dh_branch. This
    // needs to be updated accordingly.  The current code is just
    // place-holer to avoid compilation errors.
    static PyObject * moose_ObjId_getFieldIndex(_ObjId * self)
    {
        PyObject * ret = Py_BuildValue("I", self->oid_.dataId.value());
        return ret;
    }
    
    ///////////////////////////////////////////////
    // Python method lists for PyObject of ObjId
    ///////////////////////////////////////////////

    static PyMethodDef ObjIdMethods[] = {
        {"__init__", (PyCFunction)moose_ObjId_init, METH_KEYWORDS,
         moose_ObjId_init_documentation},
        {"__cmp__", (PyCFunction)moose_ObjId_compare, METH_VARARGS,
         moose_ObjId_compare_documentation},
        {"getFieldType", (PyCFunction)moose_ObjId_getFieldType, METH_VARARGS,
         moose_ObjId_getFieldType_documentation},        
        {"getField", (PyCFunction)moose_ObjId_getField, METH_VARARGS,
         moose_ObjId_getField_documentation},
        {"setField", (PyCFunction)moose_ObjId_setField, METH_VARARGS,
         moose_ObjId_setField_documentation},
        {"getLookupField", (PyCFunction)moose_ObjId_getLookupField, METH_VARARGS,
         moose_ObjId_getLookupField_documentation},
        {"setLookupField", (PyCFunction)moose_ObjId_setLookupField, METH_VARARGS,
         moose_ObjId_setLookupField_documentation},
        {"getId", (PyCFunction)moose_ObjId_getId, METH_NOARGS,
         moose_ObjId_getId_documentation},
        {"getFieldNames", (PyCFunction)moose_ObjId_getFieldNames, METH_VARARGS,
         moose_ObjId_getFieldNames_documenation},
        {"getNeighbors", (PyCFunction)moose_ObjId_getNeighbors, METH_VARARGS,
         moose_ObjId_getNeighbors_documentation},
        {"connect", (PyCFunction)moose_ObjId_connect, METH_VARARGS,
         moose_ObjId_connect_documentation},
        {"getDataIndex", (PyCFunction)moose_ObjId_getDataIndex, METH_NOARGS,
         moose_ObjId_getDataIndex_documentation},
        {"getFieldIndex", (PyCFunction)moose_ObjId_getFieldIndex, METH_NOARGS,
         "Get the index of this object as a field."},
        {"setDestField", (PyCFunction)moose_ObjId_setDestField, METH_VARARGS,
         moose_ObjId_setDestField_documentation},
        {NULL, NULL, 0, NULL},        /* Sentinel */        
    };

    ///////////////////////////////////////////////
    // Type defs for PyObject of ObjId
    ///////////////////////////////////////////////
    PyTypeObject ObjIdType = { 
        PyObject_HEAD_INIT(NULL)            /* tp_head */
        0,                                  /* tp_internal */
        "moose.ObjId",                      /* tp_name */
        sizeof(_ObjId),                     /* tp_basicsize */
        0,                                  /* tp_itemsize */
        0,                                  /* tp_dealloc */
        0,                                  /* tp_print */
        0,                                  /* tp_getattr */
        0,                                  /* tp_setattr */
        (cmpfunc)moose_ObjId_compare,       /* tp_compare */
        (reprfunc)moose_ObjId_repr,         /* tp_repr */
        0,                                  /* tp_as_number */
        0,                                  /* tp_as_sequence */
        0,                                  /* tp_as_mapping */
        (hashfunc)moose_ObjId_hash,         /* tp_hash */
        0,                                  /* tp_call */
        (reprfunc)moose_ObjId_repr,         /* tp_str */
        (getattrofunc)moose_ObjId_getattro, /* tp_getattro */
        (setattrofunc)moose_ObjId_setattro, /* tp_setattro */
        0,                                  /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        "Individual moose object contained in an array-type object.",
        0,                                  /* tp_traverse */
        0,                                  /* tp_clear */
        0,                                  /* tp_richcompare */
        0,                                  /* tp_weaklistoffset */
        0,                                  /* tp_iter */
        0,                                  /* tp_iternext */
        ObjIdMethods,                       /* tp_methods */
        0,                                  /* tp_members */
        0,                                  /* tp_getset */
        0,                                  /* tp_base */
        0,                                  /* tp_dict */
        0,                                  /* tp_descr_get */
        0,                                  /* tp_descr_set */
        0,                                  /* tp_dictoffset */
        (initproc) moose_ObjId_init,        /* tp_init */
        0,                                  /* tp_alloc */
        0,                                  /* tp_new */
        0,                                  /* tp_free */
    };

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

    static PyObject * moose_getFieldNames(PyObject * dummy, PyObject * args)
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
                 "src : Id, ObjId or str\n"
                 "\tsource object.\n"
                 "dest : Id, ObjId or str\n"
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
                 "True on success, False on failure\n"
                 );
    static PyObject * moose_copy(PyObject * dummy, PyObject * args, PyObject * kwargs)
    {
        PyObject * src = NULL, * dest = NULL;
        char * newName = NULL;
        static const char * kwlist[] = {"src", "dest", "name", "n", "toGlobal", "copyExtMsg", NULL};
        unsigned int num=1, toGlobal=0, copyExtMsgs=0;
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|sIII", const_cast<char**>(kwlist), &src, &dest, &newName, &num, &toGlobal, &copyExtMsgs)){
            return NULL;
        }
        Id _src, _dest;
        if (Id_SubtypeCheck(src)){
            _src = ((_Id*)src)->id_;
        } else if (ObjId_SubtypeCheck(src)){
            _src = ((_ObjId*)src)->oid_.id;
        } else if (PyString_Check(src)){
            _src = Id(PyString_AsString(src));
        } else {
            PyErr_SetString(PyExc_TypeError, "Source must be instance of Id, ObjId or string.");
            return NULL;
        }
        if (_src == Id()){
            PyErr_SetString(PyExc_ValueError, "Cannot make copy of moose shell.");
            return NULL;
        } 
        if (Id_SubtypeCheck(dest)){
            _dest = ((_Id*)dest)->id_;
        } else if (ObjId_SubtypeCheck(dest)){
            _dest = ((_ObjId*)dest)->oid_.id;
        } else if (PyString_Check(dest)){
            _dest = Id(PyString_AsString(dest));
        } else {
            PyErr_SetString(PyExc_TypeError, "Destination must be instance of Id, ObjId or string.");
            return NULL;
        }
        string name;
        if (newName == NULL){
            // Use the original name if name is not specified.
            name = Field<string>::get(ObjId(_src, 0), "name");
        } else {
            name = string(newName);
        }
        _Id * tgt = PyObject_New(_Id, &IdType);
        tgt->id_ = ShellPtr->doCopy(_src, _dest, name, num, toGlobal, copyExtMsgs);
        PyObject * ret = (PyObject*)tgt;
        return ret;
    }

    // Not sure what this function should return... ideally the Id of the
    // moved object - does it change though?
    static PyObject * moose_move(PyObject * dummy, PyObject * args)
    {
        PyObject * src, * dest;
        if (!PyArg_ParseTuple(args, "OO:moose_move", &src, &dest)){
            return NULL;
        }
        if (((_Id*)src)->id_ == Id()){
            PyErr_SetString(PyExc_ValueError, "Cannot move moose shell");
            return NULL;
        }
        ShellPtr->doMove(((_Id*)src)->id_, ((_Id*)dest)->id_);
        Py_RETURN_NONE;
    }

    static PyObject * moose_delete(PyObject * dummy, PyObject * args)
    {
        PyObject * obj;
        if (!PyArg_ParseTuple(args, "O:moose_delete", &obj)){
            return NULL;
        }
        if (((_Id*)obj)->id_ == Id()){
            PyErr_SetString(PyExc_ValueError, "Cannot delete moose shell.");
            return NULL;
        }
        ShellPtr->doDelete(((_Id*)obj)->id_);
        ((_Id*)obj)->id_ = Id();
        Py_RETURN_NONE;
    }

    static PyObject * moose_useClock(PyObject * dummy, PyObject * args)
    {
        char * path, * field;
        unsigned int tick;
        if(!PyArg_ParseTuple(args, "Iss:moose_useClock", &tick, &path, &field)){
            return NULL;
        }
        ShellPtr->doUseClock(string(path), string(field), tick);
        Py_RETURN_NONE;
    }
    static PyObject * moose_setClock(PyObject * dummy, PyObject * args)
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
        ShellPtr->doSetClock(tick, dt);
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
                 "Parameters\n"
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
    static PyObject * moose_start(PyObject * dummy, PyObject * args)
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
        ShellPtr->doStart(runtime);
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
    static PyObject * moose_reinit(PyObject * dummy, PyObject * args)
    {
        ShellPtr->doReinit();
        Py_RETURN_NONE;
    }
    static PyObject * moose_stop(PyObject * dummy, PyObject * args)
    {
        ShellPtr->doStop();
        Py_RETURN_NONE;
    }
    static PyObject * moose_isRunning(PyObject * dummy, PyObject * args)
    {
        return Py_BuildValue("i", ShellPtr->isRunning());
    }

    static PyObject * moose_exists(PyObject * dummy, PyObject * args)
    {
        char * path;
        if (!PyArg_ParseTuple(args, "s", &path)){
            return NULL;
        }
        return Py_BuildValue("i", Id(path) != Id() || string(path) == "/" || string(path) == "/root");
    }

    //Harsha : For writing genesis file to sbml
    static PyObject * moose_writeSBML(PyObject * dummy, PyObject * args)
    {
        char * fname = NULL, * modelpath = NULL;
        if(!PyArg_ParseTuple(args, "ss:moose_writeSBML", &fname, &modelpath)){
            return NULL;
        }
        
        int ret = ShellPtr->doWriteSBML(string(fname), string(modelpath));
        return Py_BuildValue("i", ret);
    }
    PyDoc_STRVAR(moose_loadModel_documentation,
                 "loadModel(filename, modelpath, solverclass) -> moose.Id\n"
                 "\n"
                 "Load model from a file to a specified path.\n"
                 "\n"
                 "Parameters\n"
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
                 "Id instance refering to the loaded model container.\n"
                 );
                 static PyObject * moose_loadModel(PyObject * dummy, PyObject * args)
    {
        char * fname = NULL, * modelpath = NULL, * solverclass = NULL;
        if(!PyArg_ParseTuple(args, "ss|s:moose_loadModel", &fname, &modelpath, &solverclass)){
            return NULL;
        }
        _Id * model = (_Id*)PyObject_New(_Id, &IdType);
        if (!solverclass){
            model->id_ = ShellPtr->doLoadModel(string(fname), string(modelpath));
        } else {
            model->id_ = ShellPtr->doLoadModel(string(fname), string(modelpath), string(solverclass));
        }
        PyObject * ret = reinterpret_cast<PyObject*>(model);
        return ret;
    }

    static PyObject * moose_setCwe(PyObject * dummy, PyObject * args)
    {
        PyObject * element = NULL;
        const char * path = "/";
        Id id;
        if(PyArg_ParseTuple(args, "s:moose_setCwe", const_cast<char**>(&path))){
            id = Id(string(path));
        } else if (PyArg_ParseTuple(args, "O:moose_setCwe", &element)){
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
        ShellPtr->setCwe(id);
        Py_RETURN_NONE;
    }

    static PyObject * moose_getCwe(PyObject * dummy, PyObject * args)
    {
        if (!PyArg_ParseTuple(args, ":moose_getCwe")){
            return NULL;
        }
        _Id * cwe = (_Id*)PyObject_New(_Id, &IdType);
        cwe->id_ = ShellPtr->getCwe();        
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
                 "src : ObjId\n"
                 "\tthe source object\n"
                 "src_field : str\n"
                 "\tthe source field name. Fields listed under `srcFinfo` and\n"
                 "`sharedFinfo` qualify for this.\n"
                 "dest : ObjId\n"
                 "\tthe destination object.\n"
                 "dest_field : str\n"
                 "\tthe destination field name. Fields listed under `destFinfo`\n"
                 "and `sharedFinfo` qualify for this.\n"
                 "message_type : str (optional)\n"
                 "\tType of the message. Can be `Single`, `OneToOne`, `OneToAll`.\n"
                 "If not specified, it defaults to `Single`.\n"
                 "\n"
                 "Example\n"
                 "-------\n"
                 "Connect the output of a pulse generator to the input of a spike\n"
                 "generator:\n"
                 "\n"
                 ">>> pulsegen = moose.PulseGen('pulsegen')\n"
                 ">>> spikegen = moose.SpikeGen('spikegen')\n"
                 ">>> moose.connect(pulsegen, 'outputOut', spikegen, 'Vm')\n"
                 "1\n"
                 "\n"

                 );
    static PyObject * moose_connect(PyObject * dummy, PyObject * args)
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
        _ObjId * dest = reinterpret_cast<_ObjId*>(destPtr);
        _ObjId * src = reinterpret_cast<_ObjId*>(srcPtr);
        bool ret = (ShellPtr->doAddMsg(msgType, src->oid_, string(srcField), dest->oid_, string(destField)) != Msg::bad);
        if (!ret){
            PyErr_SetString(PyExc_NameError, "connect failed: check field names and type compatibility.");
            return NULL;
        }
        return Py_BuildValue("i", ret);        
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
                 ">>> moose.getFieldDict('Neutral', 'srcFinfo')\n"
                 "{'childMsg': 'int'}\n"
                 "\n");
    static PyObject * moose_getFieldDict(PyObject * dummy, PyObject * args)
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

    static PyObject * moose_getField(PyObject * dummy, PyObject * args)
    {
        PyObject * pyobj;
        const char * field;
        const char * type;
        if (!PyArg_ParseTuple(args, "Oss:moose_getfield", &pyobj, &field, &type)){
            return NULL;
        }
        if (!ObjId_SubtypeCheck(pyobj)){
            PyErr_SetString(PyExc_TypeError, "moose.getField(ObjId, fieldname, fieldtype): First argument must be an instance of ObjId or its subclass");
            return NULL;
        }
        string fname(field), ftype(type);
        ObjId oid = ((_ObjId*)pyobj)->oid_;
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
                    Py_XDECREF(entry);
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
                    Py_XDECREF(entry);
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
                    Py_XDECREF(entry);
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
                    Py_XDECREF(entry);
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
                    Py_XDECREF(entry);
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
                    Py_XDECREF(entry);
                    return NULL;                                 
                }                                            
            }
            return ret;
        }
        PyErr_SetString(PyExc_TypeError, "Field type not handled.");
        return NULL;
    }

    static PyObject * moose_syncDataHandler(PyObject * dummy, _Id * target)
    {
        ShellPtr->doSyncDataHandler(target->id_);
        Py_RETURN_NONE;
    }


    static PyObject * moose_seed(PyObject * dummy, PyObject * args)
    {
        long seed = 0;
        if (!PyArg_ParseTuple(args, "|l", &seed)){
            return NULL;
        }
        mtseed(seed);
        Py_RETURN_NONE;
    }

    static PyObject * moose_wildcardFind(PyObject * dummy, PyObject * args)
    {
        vector <Id> objects;
        char * wildcard_path = NULL;
        if (!PyArg_ParseTuple(args, "s:moose.wildcardFind", &wildcard_path)){
            return NULL;
        }
        ShellPtr->wildcard(string(wildcard_path), objects);
        PyObject * ret = PyTuple_New(objects.size());
        if (ret == NULL){
            PyErr_SetString(PyExc_RuntimeError, "moose.wildcardFind: failed to allocate new tuple.");
            return NULL;
        }
            
        for (unsigned int ii = 0; ii < objects.size(); ++ii){
            _Id * entry = PyObject_New(_Id, &IdType);                       
            if (!entry){
                Py_XDECREF(ret);
                PyErr_SetString(PyExc_RuntimeError, "moose.wildcardFind: failed to allocate new Id.");
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
    static PyObject * moose_quit(PyObject * dummy)
    {
        finalize();
        cout << "Quitting MOOSE." << endl;
        Py_RETURN_NONE;
    }

    /// Go through all elements under /classes and ask for defining a
    /// Python class for it.
    static int defineAllClasses(PyObject * module_dict)
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

    static int define_class(PyObject * module_dict, const Cinfo * cinfo)
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
        Py_TYPE(new_class) = &PyType_Type;
        new_class->tp_flags = Py_TPFLAGS_DEFAULT |
                Py_TPFLAGS_BASETYPE;
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
        new_class->ob_size = sizeof(_ObjId);        
        string str = "moose." + class_name;
        new_class->tp_name = (char *)calloc(str.length()+1,
                                            sizeof(char));
        strncpy(const_cast<char*>(new_class->tp_name), str.c_str(),
                str.length());
        new_class->tp_doc = moose_Class_documentation;
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
        PyDict_SetItemString(new_class->tp_dict, "__module__", PyString_FromString("moose"));
        string doc = const_cast<Cinfo*>(cinfo)->getDocs();
        PyDict_SetItemString(new_class->tp_dict, "__doc__", PyString_FromString(doc.c_str()));
        PyDict_SetItemString(module_dict, class_name.c_str(), (PyObject *)new_class);
        return 1;                
    }
    
    static PyObject * moose_ObjId_get_destField_attr(PyObject * self, void * closure)
    {
        if (!ObjId_SubtypeCheck(self)){
            PyErr_SetString(PyExc_TypeError, "First argument must be an instance of ObjId");
            return NULL;
        }
        char * name = NULL;
        if (!PyArg_ParseTuple((PyObject *)closure,
                              "s:_get_destField: "
                              "expected a string in getter closure.",
                              &name)){
            return NULL;
        }
        // If the DestField already exists, return it
        string full_name = ((_ObjId*)self)->oid_.path() +
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
            Py_XINCREF(ret);
            get_inited_destfields()[full_name] =  (PyObject*)ret;
            return (PyObject*)ret;
        }
        Py_XDECREF((PyObject*)ret);
        Py_XDECREF(args);
        return NULL;
    }
    
    
    static int define_destFinfos(const Cinfo * cinfo)
    {
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
            // vec[curr_index].doc = ;
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

    static PyObject * moose_ObjId_get_lookupField_attr(PyObject * self,
                                                       void * closure)
    {
        if (!ObjId_SubtypeCheck(self)){
            PyErr_SetString(PyExc_TypeError,
                            "First argument must be an instance of ObjId");
            return NULL;
        }
        char * name = NULL;
        if (!PyArg_ParseTuple((PyObject *)closure,
                              "s:moose_ObjId_get_lookupField_attr: expected a string in getter closure.",
                              &name)){
            return NULL;
        }
        assert(name);
        // If the LookupField already exists, return it
        string full_name = ((_ObjId*)self)->oid_.path() + "." + string(name);
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
    
    static int define_lookupFinfos(const Cinfo * cinfo)
    {
        const string & class_name = cinfo->name();
        unsigned int num_lookupFinfos = cinfo->getNumLookupFinfo();
        unsigned int curr_index = get_getsetdefs()[class_name].size();
        for (unsigned int ii = 0; ii < num_lookupFinfos; ++ii){
            const string& lookupFinfo_name = const_cast<Cinfo*>(cinfo)->getLookupFinfo(ii)->name();
            PyGetSetDef getset;
            get_getsetdefs()[class_name].push_back(getset);
            get_getsetdefs()[class_name][curr_index].name = (char*)calloc(lookupFinfo_name.size() + 1, sizeof(char));
            strncpy(get_getsetdefs()[class_name][curr_index].name, const_cast<char*>(lookupFinfo_name.c_str()), lookupFinfo_name.size());
            // get_getsetdefs()[class_name][curr_index].doc = "";
            get_getsetdefs()[class_name][curr_index].get = (getter)moose_ObjId_get_lookupField_attr;
            PyObject * args = PyTuple_New(1);
            PyTuple_SetItem(args, 0, PyString_FromString(lookupFinfo_name.c_str()));
            get_getsetdefs()[class_name][curr_index].closure = (void*)args;
            ++curr_index;
        }
        return 1;
    }

    static PyObject * moose_ObjId_get_elementField_attr(PyObject * self,
                                                       void * closure)
    {
        if (!ObjId_SubtypeCheck(self)){
            PyErr_SetString(PyExc_TypeError,
                            "First argument must be an instance of ObjId");
            return NULL;
        }
        char * name = NULL;
        if (!PyArg_ParseTuple((PyObject *)closure,
                              "s:moose_ObjId_get_lookupField_attr: expected a string in getter closure.",
                              &name)){
            return NULL;
        }
        // If the ElementField already exists, return it
        string full_name = ((_ObjId*)self)->oid_.path() + "." + string(name);
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

    static int define_elementFinfos(const Cinfo * cinfo)
    {
        const string & class_name = cinfo->name();
        unsigned int num_fieldElementFinfo = cinfo->getNumFieldElementFinfo();
        unsigned int curr_index = get_getsetdefs()[class_name].size();
        for (unsigned int ii = 0; ii < num_fieldElementFinfo; ++ii){
            const string& finfo_name = const_cast<Cinfo*>(cinfo)->getFieldElementFinfo(ii)->name();
            PyGetSetDef getset;
            get_getsetdefs()[class_name].push_back(getset);
            get_getsetdefs()[class_name][curr_index].name = (char*)calloc(finfo_name.size() + 1, sizeof(char));
            strncpy(get_getsetdefs()[class_name][curr_index].name, const_cast<char*>(finfo_name.c_str()), finfo_name.size());
            // get_getsetdefs()[class_name][curr_index].doc = EMPTY_STRING;
            get_getsetdefs()[class_name][curr_index].get = (getter)moose_ObjId_get_elementField_attr;
            PyObject * args = PyTuple_New(1);
            PyTuple_SetItem(args, 0, PyString_FromString(finfo_name.c_str()));
            get_getsetdefs()[class_name][curr_index].closure = (void*)args;
            ++curr_index;
        }
        return 1;
    }

    PyObject * moose_ElementField_getNum(_Field * self, void * closure)
    {
        unsigned int num = Field<unsigned int>::get(self->owner, "num_" + string(self->name));
        return Py_BuildValue("I", num);
    }

    int moose_ElementField_setNum(_Field * self, PyObject * args, void * closure)
    {
        unsigned int num;
        if (!PyInt_Check(args)){
            PyErr_SetString(PyExc_TypeError, "moose.ElementField.setNum - needes an integer.");
            return -1;
        }
        num = PyInt_AsUnsignedLongMask(args);
        if (!Field<unsigned int>::set(self->owner, "num_" + string(self->name), num)){
            PyErr_SetString(PyExc_RuntimeError, "moose.ElementField.setNum : Field::set returned False.");
            return -1;
        }
        return 0;
    }

    PyObject * moose_ElementField_getItem(_Field * self, Py_ssize_t index)
    {
        unsigned int len = Field<unsigned int>::get(self->owner, "num_" + string(self->name));
        if (index >= len){
            PyErr_SetString(PyExc_IndexError, "moose.ElementField.getItem: index out of bounds.");
            return NULL;
        }
        if (index < 0){
            index += len;
        }
        if (index < 0){
            PyErr_SetString(PyExc_IndexError, "moose.ElementField.getItem: invalid index.");
            return NULL;
        }
        _ObjId * oid = PyObject_New(_ObjId, &ObjIdType);
        oid->oid_ = ObjId(Id(self->owner.path() + "/" + self->name), DataId(index));
        return (PyObject*)oid;
    }


    /////////////////////////////////////////////////////////////////////
    // Method definitions for MOOSE module
    /////////////////////////////////////////////////////////////////////    
    static PyMethodDef MooseMethods[] = {
        {"getFieldNames", (PyCFunction)moose_getFieldNames, METH_VARARGS, moose_getFieldNames_documentation},
        {"copy", (PyCFunction)moose_copy, METH_VARARGS|METH_KEYWORDS, moose_copy_documentation},
        {"move", (PyCFunction)moose_move, METH_VARARGS, "Move a Id object to a destination."},
        {"delete", (PyCFunction)moose_delete, METH_VARARGS, "Delete the moose object."},
        {"useClock", (PyCFunction)moose_useClock, METH_VARARGS, "Schedule objects on a specified clock"},
        {"setClock", (PyCFunction)moose_setClock, METH_VARARGS, "Set the dt of a clock."},
        {"start", (PyCFunction)moose_start, METH_VARARGS, moose_start_documentation},
        {"reinit", (PyCFunction)moose_reinit, METH_VARARGS, moose_reinit_documentation},
        {"stop", (PyCFunction)moose_stop, METH_VARARGS, "Stop simulation"},
        {"isRunning", (PyCFunction)moose_isRunning, METH_VARARGS, "True if the simulation is currently running."},
        {"exists", (PyCFunction)moose_exists, METH_VARARGS, "True if there is an object with specified path."},
        {"writeSBML", (PyCFunction)moose_writeSBML, METH_VARARGS, "Export biochemical model to an SBML file."},    
        {"loadModel", (PyCFunction)moose_loadModel, METH_VARARGS, moose_loadModel_documentation},
        {"connect", (PyCFunction)moose_connect, METH_VARARGS, moose_connect_documentation},        
        {"getCwe", (PyCFunction)moose_getCwe, METH_VARARGS, "Get the current working element. 'pwe' is an alias of this function."},
        // {"pwe", (PyCFunction)moose_getCwe, METH_VARARGS, "Get the current working element. 'getCwe' is an alias of this function."},
        {"setCwe", (PyCFunction)moose_setCwe, METH_VARARGS, "Set the current working element. 'ce' is an alias of this function"},
        // {"ce", (PyCFunction)moose_setCwe, METH_VARARGS, "Set the current working element. setCwe is an alias of this function."},
        {"getFieldDict", (PyCFunction)moose_getFieldDict, METH_VARARGS, moose_getFieldDict_documentation},
        {"getField", (PyCFunction)moose_getField, METH_VARARGS,
         "getField(ObjId, field, fieldtype) -- Get specified field of specified type from object Id."},
        {"syncDataHandler", (PyCFunction)moose_syncDataHandler, METH_VARARGS,
         "synchronizes fieldDimension on the DataHandler"
         " across nodes. Used after function calls that might alter the"
         " number of Field entries in the table."
         " The target is the FieldElement whose fieldDimension needs updating."},
        {"seed", (PyCFunction)moose_seed, METH_VARARGS, "seed(seedvalue) -- Seed the random number generator of MOOSE."},
        {"wildcardFind", (PyCFunction)moose_wildcardFind, METH_VARARGS, "Return a list of Ids by a wildcard query."},
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
    
                 
    PyMODINIT_FUNC init_moose()
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
        create_shell(argc, argv);
        for (int ii = 1; ii < argc; ++ii){
            free(argv[ii]);
        }
        // Now initialize the module
        char moose_err[] = "moose.error";
        PyObject *moose_module = Py_InitModule3("_moose",
                                                MooseMethods,
                                                moose_module_documentation);
        if (moose_module == NULL){
            return;
        }

        int registered = Py_AtExit(&finalize);
        if (registered != 0){
            cerr << "Failed to register finalize() to be called at exit. " << endl;
        }

        MooseError = PyErr_NewException(moose_err, NULL, NULL);
        if (MooseError != NULL){
            Py_INCREF(MooseError);
            PyModule_AddObject(moose_module, "error", MooseError);
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
        PyModule_AddObject(moose_module, "Id", (PyObject*)&IdType);

        // Add ObjId type
        // Py_TYPE(&ObjIdType) = &PyType_Type; // unnecessary - filled in by PyType_Ready
        ObjIdType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&ObjIdType) < 0){
            PyErr_Print();
            exit(-1);
        };
        Py_INCREF(&ObjIdType);
        PyModule_AddObject(moose_module, "ObjId", (PyObject*)&ObjIdType);

        // Add LookupField type
        // Py_TYPE(&moose_LookupField) = &PyType_Type;  // unnecessary - filled in by PyType_Ready        
        // moose_LookupField.tp_new = PyType_GenericNew;
        if (PyType_Ready(&moose_LookupField) < 0){
            PyErr_Print();
            exit(-1);
        }        
        Py_INCREF(&moose_LookupField);
        PyModule_AddObject(moose_module, "ElementField", (PyObject*)&moose_ElementField);

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
        
        PyModule_AddIntConstant(moose_module, "SINGLETHREADED", isSingleThreaded);
        PyModule_AddIntConstant(moose_module, "NUMCORES", numCores);
        PyModule_AddIntConstant(moose_module, "NUMNODES", numNodes);
        PyModule_AddIntConstant(moose_module, "MYNODE", myNode);
        PyModule_AddIntConstant(moose_module, "INFINITE", isInfinite);
        PyModule_AddStringConstant(moose_module, "__version__", ShellPtr->doVersion().c_str());
        PyModule_AddStringConstant(moose_module, "VERSION", ShellPtr->doVersion().c_str());
        PyModule_AddStringConstant(moose_module, "SVN_REVISION", ShellPtr->doRevision().c_str());
        PyObject * module_dict = PyModule_GetDict(moose_module);
        clock_t defclasses_start = clock();
        if (!defineAllClasses(module_dict)){
            PyErr_Print();
            exit(-1);
        }
        clock_t defclasses_end = clock();
        cout << "Info: Time to define moose classes:" << (defclasses_end - defclasses_start) * 1.0 /CLOCKS_PER_SEC << endl;
        PyGILState_Release(gstate);
        clock_t modinit_end = clock();
        cout << "Info: Time to initialize module:" << (modinit_end - modinit_start) * 1.0 /CLOCKS_PER_SEC << endl;

    } //! init_moose
    
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
