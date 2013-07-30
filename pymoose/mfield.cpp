// mfield.cpp --- 
// 
// Filename: mfield.cpp
// Description: 
// Author: 
// Maintainer: 
// Created: Mon Jul 22 17:03:03 2013 (+0530)
// Version: 
// Last-Updated: Tue Jul 23 16:11:50 2013 (+0530)
//           By: subha
//     Update #: 13
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
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
// 
// 

// Code:

#include <Python.h>
#include <structmember.h> // This defines the type id macros like T_STRING
#include "numpy/arrayobject.h"

#include <iostream>
#include <typeinfo>
#include <cstring>
#include <map>
#include <ctime>

#include "../basecode/header.h"
#include "../basecode/Id.h"
#include "../basecode/ObjId.h"
#include "../basecode/DataId.h"
#include "../utility/utility.h"
#include "../randnum/randnum.h"
#include "../shell/Shell.h"

#include "moosemodule.h"

using namespace std;

extern "C" {

    extern PyTypeObject ObjIdType;
    extern PyTypeObject IdType;

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
    int moose_Field_init(_Field * self, PyObject * args, PyObject * kwargs)
    {
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
        if (!PyObject_IsInstance(owner, (PyObject*)&ObjIdType)){
            PyErr_SetString(PyExc_TypeError, "Owner must be subtype of ObjId");
            return -1;
        }
        self->owner = ((_ObjId*)owner)->oid_;
        if (!Id::isValid(self->owner.id)){
            Py_XDECREF(owner);
            Py_XDECREF(self);
            RAISE_INVALID_ID(-1, "moose_Field_init");
        }
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
    long moose_Field_hash(_Field * self)
    {
        if (!Id::isValid(self->owner.id)){
            RAISE_INVALID_ID(-1, "moose_Field_hash");
        }
        string fieldPath = self->owner.path() + "." + self->name;
        PyObject * path = PyString_FromString(fieldPath.c_str());
        long hash = PyObject_Hash(path);
        Py_XDECREF(path);
        return hash;    
    }

    /// String representation of fields is `{objectpath}.{fieldName}`
    PyObject * moose_Field_repr(_Field * self)
    {
        if (!Id::isValid(self->owner.id)){
            RAISE_INVALID_ID(NULL, "moose_Field_repr");
        }
        ostringstream fieldPath;
        fieldPath << self->owner.path() << "." << self->name;
        return PyString_FromString(fieldPath.str().c_str());
    }

    
    PyDoc_STRVAR(moose_Field_documentation,
                 "Base class for MOOSE fields.\n"
                 "\n"
                 "Instances contain the field name and a pointer to the owner\n"
                 "object. Note on hash: the Field class is hashable but the hash is\n"
                 "constructed from the path of the container element and the field\n"
                 "name. Hence changing the name of the container element will cause the\n"
                 "hash to change. This is rather unusual in a moose script, but if you\n"
                 "are putting fields as dictionary keys, you should do that after names\n"
                 "of all elements have been finalized.");


    static PyTypeObject moose_Field = {
        PyVarObject_HEAD_INIT(NULL, 0)
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

    PyObject * moose_LookupField_getItem(_Field * self, PyObject * key)
    {
        return getLookupField(self->owner, self->name, key);
    }

    int moose_LookupField_setItem(_Field * self, PyObject * key,
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
    PyTypeObject moose_LookupField = {
        PyVarObject_HEAD_INIT(NULL, 0)
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


    PyObject * moose_DestField_call(PyObject * self, PyObject * args,
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
                 

    PyTypeObject moose_DestField = {
        PyVarObject_HEAD_INIT(NULL, 0)
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
                 "example, synapse in an IntFire neuron. Element fields can be traversed\n"
                 "like a sequence. Additionally, you can set the number of entries by\n"
                 "setting the `num` attribute to a desired value.\n");

    PyDoc_STRVAR(moose_ElementField_num_documentation,
                 "Number of entries in the field.");

    PyDoc_STRVAR(moose_ElementField_path_documentation,
                 "Path of the field element.");
    PyDoc_STRVAR(moose_ElementField_id_documentation,
                 "Id of the field element.");
    static char numfield[] = "num";
    static char path[] = "path";
    static char id[] = "id_";
    static PyGetSetDef ElementFieldGetSetters[] = {
        {numfield,
         (getter)moose_ElementField_getNum,
         (setter)moose_ElementField_setNum,
         moose_ElementField_num_documentation,
         NULL},
        {path,
         (getter)moose_ElementField_getPath,
         NULL,
         moose_ElementField_path_documentation,
         NULL},
        {id,
         (getter)moose_ElementField_getId,
         NULL,
         moose_ElementField_id_documentation,
         NULL},
        {NULL}, /* sentinel */
    };
    
    static PySequenceMethods ElementFieldSequenceMethods = {
        (lenfunc)moose_ElementField_getLen, // sq_length
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

    PyTypeObject moose_ElementField = {
        PyVarObject_HEAD_INIT(NULL, 0)
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

    PyObject * moose_ElementField_getNum(_Field * self, void * closure)
    {
        if (!Id::isValid(self->owner.id)){
            RAISE_INVALID_ID(NULL, "moose_ElementField_getNum");
        }
        unsigned int num = Field<unsigned int>::get(self->owner, "num_" + string(self->name));
        return Py_BuildValue("I", num);
    }

    Py_ssize_t moose_ElementField_getLen(_Field * self, void * closure)
    {
        if (!Id::isValid(self->owner.id)){
            RAISE_INVALID_ID(-1, "moose_ElementField_getLen");
        }
        unsigned int num = Field<unsigned int>::get(self->owner, "num_" + string(self->name));
        return Py_ssize_t(num);
    }

    int moose_ElementField_setNum(_Field * self, PyObject * args, void * closure)
    {
        if (!Id::isValid(self->owner.id)){
            RAISE_INVALID_ID(-1, "moose_ElementField_setNum");
        }
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

    PyObject * moose_ElementField_getPath(_Field * self, void * closure)
    {
        if (!Id::isValid(self->owner.id)){
            RAISE_INVALID_ID(NULL, "moose_ElementField_setNum");
        }
        string path = Id(self->owner.path() + "/" + string(self->name)).path();
        return Py_BuildValue("s", path.c_str());
    }

    PyObject * moose_ElementField_getId(_Field * self, void * closure)
    {
        if (!Id::isValid(self->owner.id)){
            RAISE_INVALID_ID(NULL, "moose_ElementField_setNum");
        }
        Id myId(self->owner.path() + "/" + string(self->name));
        _Id * new_id = PyObject_New(_Id, &IdType);
        new_id->id_ = myId;
        return (PyObject*)new_id;
    }

    PyObject * moose_ElementField_getItem(_Field * self, Py_ssize_t index)
    {
        if (!Id::isValid(self->owner.id)){
            RAISE_INVALID_ID(NULL, "moose_ElementField_getItem");
        }
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
        // cout << "Element field: " << self->name << ", owner: " << self->owner.path() << endl;
        stringstream path;
        path << self->owner.path() << "/" << self->name << "[" << index << "]";
        // cout << "moose_ElementField_getItem:: path=" << path.str();
        oid->oid_ = ObjId(path.str());
        return (PyObject*)oid;
    }
    
} // extern "C"


// 
// mfield.cpp ends here
