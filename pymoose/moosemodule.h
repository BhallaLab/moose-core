// moosemodule.h --- 
// 
// Filename: moosemodule.h
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Thu Mar 10 17:11:06 2011 (+0530)
// Version: 
// Last-Updated: Wed Mar 30 15:49:11 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 239
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
        Id _id;
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
        ObjId _id;
    } _ObjId;

    static PyObject * MooseError;
    //////////////////////////////////////////
    // Methods for Id class
    //////////////////////////////////////////
    static int _pymoose_Id_init(_Id * self, PyObject * args, PyObject * kwargs);
    static void _pymoose_Id_dealloc(_Id * self);
    static PyObject * _pymoose_Id_repr(_Id * self);
    static PyObject * _pymoose_Id_str(_Id * self);
    static PyObject * _pymoose_Id_destroy(_Id * self, PyObject * args);
    static PyObject * _pymoose_Id_getId(_Id * self, PyObject * args);
    static PyObject * _pymoose_Id_getPath(_Id * self, PyObject * args);
    static PyObject * _pymoose_Id_syncDataHandler(_Id * self, PyObject * args);
    /* Id functions to allow part of sequence protocol */
    static Py_ssize_t _pymoose_Id_getLength(_Id * self);
    static PyObject * _pymoose_Id_getItem(_Id * self, Py_ssize_t index);
    static PyObject * _pymoose_Id_getSlice(_Id * self, PyObject * args);    
    static PyObject * _pymoose_Id_getShape(_Id * self, PyObject * args);    
    static int _pymoose_Id_richCompare(_Id * self, PyObject * args, int op);
    static int _pymoose_Id_contains(_Id * self, PyObject * args);
    ///////////////////////////////////////////
    // Methods for ObjId class
    ///////////////////////////////////////////
    static int _pymoose_ObjId_init(_ObjId * self, PyObject * args, PyObject * kwargs);
    static void _pymoose_ObjId_dealloc(_ObjId * self);
    static PyObject * _pymoose_ObjId_repr(_ObjId * self);
    static PyObject * _pymoose_ObjId_str(_ObjId * self);
    static PyObject * _pymoose_ObjId_getField(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_setField(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getFieldNames(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getFieldType(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getDataIndex(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getFieldIndex(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_getId(_ObjId * self, PyObject * args);
    static PyObject * _pymoose_ObjId_connect(_ObjId * self, PyObject * args);
    static int _pymoose_ObjId_richCompare(_ObjId * self, PyObject * args, int op);
    
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
    static PyObject * _pymoose_loadModel(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_setCwe(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_getCwe(PyObject * dummy, PyObject * args);
    static PyObject * _pymoose_copy(PyObject * dummy, PyObject * args, PyObject * kwargs);
    static PyObject * _pymoose_move(PyObject * dummy, PyObject * args);

    PyMODINIT_FUNC init_moose();
} //!extern "C"


#endif // _MOOSEMODULE_H

// 
// moosemodule.h ends here
