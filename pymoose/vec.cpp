// vec.cpp --- 
// 
// Filename: vec.cpp
// Description: 
// Author: 
// Maintainer: 
// Created: Mon Jul 22 16:46:37 2013 (+0530)
// Version: 
// Last-Updated: Tue Jul 23 19:10:11 2013 (+0530)
//           By: subha
//     Update #: 58
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
// Mon Jul 22 16:47:10 IST 2013 - Splitting contents of
// moosemodule.cpp into speparate files.
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

extern int verbosity;
extern "C" {
    ///////////////////////////////////////////////
    // Python method lists for PyObject of Id
    ///////////////////////////////////////////////
    PyDoc_STRVAR(moose_Id_delete_doc,
                 "vec.delete()"
                 "\n"
                 "\nDelete the underlying moose object. This will invalidate all"
                 "\nreferences to this object and any attempt to access it will raise a"
                 "\nValueError."
                 "\n");

    PyDoc_STRVAR(moose_Id_setField_doc,
                 "setField(fieldname, value_vector)\n"
                 "\n"
                 "Set the value of `fieldname` in all elements under this vec.\n"
                 "\n"
                 "Parameters\n"
                 "----------\n"
                 "fieldname: str\n"
                 "\tfield to be set.\n"
                 "value: sequence of values\n"
                 "\tsequence of values corresponding to individual elements under this\n"
                 "vec.\n"
                 "\n"
                 "NOTE: This is an interface to SetGet::setVec\n"
                 );
    
    static PyMethodDef IdMethods[] = {
        // {"init", (PyCFunction)moose_Id_init, METH_VARARGS,
        //  "Initialize a Id object."},
        {"delete", (PyCFunction)moose_Id_delete, METH_NOARGS,
         moose_Id_delete_doc},
        {"getValue", (PyCFunction)moose_Id_getValue, METH_NOARGS,
         "Return integer representation of the id of the element."},
        {"getPath", (PyCFunction)moose_Id_getPath, METH_NOARGS,
         "Return the path of this vec object."},
        {"getShape", (PyCFunction)moose_Id_getShape, METH_NOARGS,
         "Get the shape of the vec object as a tuple."},
        {"setField", (PyCFunction)moose_Id_setField, METH_VARARGS,
         moose_Id_setField_doc},
        {NULL, NULL, 0, NULL},        /* Sentinel */        
    };

    static PySequenceMethods IdSequenceMethods = {
        (lenfunc)moose_Id_getLength, // sq_length
        0, //sq_concat
        0, //sq_repeat
        (ssizeargfunc)moose_Id_getItem, //sq_item
#ifndef PY3K
        (ssizessizeargfunc)moose_Id_getSlice, // getslice
#endif
        0, //sq_ass_item
#ifndef PY3K
        0, // setslice
#endif
        (objobjproc)moose_Id_contains, // sq_contains
        0, // sq_inplace_concat
        0 // sq_inplace_repeat
    };

    static PyMappingMethods IdMappingMethods = {
        (lenfunc)moose_Id_getLength, //mp_length
        (binaryfunc)moose_Id_subscript, // mp_subscript
        0 // mp_ass_subscript
    };

    ///////////////////////////////////////////////
    // Type defs for PyObject of Id
    ///////////////////////////////////////////////

    PyDoc_STRVAR(moose_Id_doc,
                 "An object uniquely identifying a moose element. moose elements are"
                 "\narray-like objects which can have one or more single-objects within"
                 "\nthem. vec can be traversed like a Python sequence and is item is an"
                 "\nelement identifying single-objects contained in the array element."
                 "\n"
                 "\nField access to ematrices are vectorized. For example, vec.name returns a"
                 "\ntuple containing the names of all the single-elements in this"
                 "\nvec. There are a few special fields that are unique for vec and are not"
                 "\nvectorized. These are `path`, `value`, `shape` and `className`."
                 "\nThere are two ways an vec can be initialized, (1) create a new array"
                 "\nelement or (2) create a reference to an existing object."
                 "\n"
                 "\n__init__(self, path=path, dims=dimesions, dtype=className)"
                 "\n"
                 "\nParameters"
                 "\n----------"                 
                 "\npath : str "
                 "\nPath of an existing array element or for creating a new one. This has"
                 "\nthe same format as unix file path: /{element1}/{element2} ... If there"
                 "\nis no object with the specified path, moose attempts to create a new"
                 "\narray element. For that to succeed everything until the last `/`"
                 "\ncharacter must exist or an error is raised"
                 "\n"
                 "\nn : positive int"
                 "\nThis is a positive integers specifying the size of the array element"
                 "\nto be created. Thus n=2 will create an"
                 "\nvec with 2 elements."
                 "\n"
                 "\n__init__(self, id)"
                 "\n"
                 "\nCreate a reference to an existing array object."
                 "\n"
                 "\nParameters"
                 "\n----------"
                 "\nid : vec/int"
                 "\nvec of an existing array object. The new object will be another"
                 "\nreference to this object."
                 "\n"
                 );
    
    PyTypeObject IdType = { 
        PyVarObject_HEAD_INIT(NULL, 0)               /* tp_head */
        "moose.vec",                  /* tp_name */
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
        &IdMappingMethods,              /* tp_as_mapping */
        (hashfunc)moose_Id_hash,                                  /* tp_hash */
        0,                                  /* tp_call */
        (reprfunc)moose_Id_str,               /* tp_str */
        (getattrofunc)moose_Id_getattro,            /* tp_getattro */
        (setattrofunc)moose_Id_setattro,            /* tp_setattro */
        0,                                  /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        moose_Id_doc,
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

    
    extern PyTypeObject ObjIdType;

    //////////////////////////////////////////////////
    // Id functions
    //////////////////////////////////////////////////

    PyObject* get_Id_attr(_Id * id, string attribute)
    {
        if (attribute == "path"){
            return moose_Id_getPath(id);
        } else if (attribute == "name"){
            string name = Field<string>::get(id->id_, "name");
            return Py_BuildValue("s", name.c_str());
        } else if (attribute == "value"){
            return moose_Id_getValue(id);
        } else if (attribute == "shape"){
            return moose_Id_getShape(id);
        } else if (attribute == "className"){
            // !NOTE: Subha: 2012-08-20 19:52:21 (+0530) - the second
            // !check is to catch a strange bug where the field passed
            // !to moose_Id_getattro is 'class' in stead of
            // !'class_'. Need to figure out how it is happening.
            // !Subha: 2012-08-21 13:25:06 (+0530) It turned out to be
            // !a GCC optimization issue. GCC was optimizing the call
            // !to get_field_alias by directly replacing `class_` with
            // !`class`. This optimized run-path was somehow being
            // !used only when therewas access to
            // !obj.parent.class_. Possibly some form of cache
            // !optimization.
            // class is a common attribute to all ObjIds under this
            // Id. Expect it to be a single value in stead of a list
            // of class names.
            
            string classname = Field<string>::get(id->id_, "className");
            return Py_BuildValue("s", classname.c_str());
        }        
        return NULL;
    }

    /**
       Utility function to create objects from full path, dimensions
       and classname.
    */
    Id create_Id_from_path(string path, unsigned int numData, unsigned int isGlobal, string type)
    {
        string parent_path;
        string name;
        string trimmed_path = trim(path);
        size_t pos = trimmed_path.rfind("/");
        if (pos != string::npos){
            name = trimmed_path.substr(pos+1);
            parent_path = trimmed_path.substr(0, pos);
        } else {
            name = trimmed_path;
        }
        // handle relative path
        if (trimmed_path[0] != '/'){
            string current_path = SHELLPTR->getCwe().path();
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
            return Id();
        }
        return SHELLPTR->doCreate(type,
                                  parent_id,
                                  string(name),
                                  numData, 
								  static_cast< NodePolicy >( isGlobal ) );
        
    }
    
     int moose_Id_init(_Id * self, PyObject * args, PyObject * kwargs)
    {
        extern PyTypeObject IdType;
        PyObject * src = NULL;
        unsigned int id = 0;
        unsigned int isGlobal = 0;
        char * type = NULL;
        // first try parsing the arguments as (path, size, classname)
        char _path[] = "path";
        char _dtype[] = "dtype";
        char _size[] = "n";
        char _global[] = "g";
        static char * kwlist[] = {_path, _size,  _global, _dtype,NULL};
        char * path = NULL;
        char _default_type[] = "Neutral";
        unsigned int numData = 0; // set it to 0 for unspecified value
        // char *type = _default_type;
        bool parse_success = false;
        /* The expected arguments are:
           string path, 
           [unsigned int numData] - default: 1
           [unsigned int isGlobal] - default: 0
           [string type] - default: Neutral
        */
        if (kwargs == NULL){
            if(PyArg_ParseTuple(args,
                                "s|IIs:moose_Id_init",
                                &path,
                                &numData,
                                &isGlobal,
                                &type)){
                parse_success = true;
            }
        } else if (PyArg_ParseTupleAndKeywords(args,
                                               kwargs,
                                               "s|IIs:moose_Id_init",
                                               kwlist,
                                               &path,
                                               &numData,
                                               &isGlobal,
                                               &type)){
            parse_success = true;
        }
        // Parsing args successful, if any error happens now,
        // different argument processing will not help. Return error
        if (parse_success){
            string trimmed_path(path);
            trimmed_path = trim(trimmed_path);
            size_t length = trimmed_path.length();
            if (length <= 0){
                PyErr_SetString(PyExc_ValueError,
                                "path must be non-empty string.");
                Py_XDECREF(self);
                return -1;
            }
            self->id_ = Id(trimmed_path);
            // Return already existing object
            if (self->id_ != Id() ||
                trimmed_path == "/" ||
                trimmed_path == "/root"){
                if ((numData > 0) && (numData != Field<unsigned int>::get(self->id_, "numData"))){
                    PyErr_WarnEx(NULL, "moose_Id_init_: Length specified does not match that of existing object.", 1);
                }
                return 0;
            }
			/**
			 * Need Subha's help here, to get rid of dims.
            vector<int> vec_dims = pysequence_to_dimvec(dims);
            if (vec_dims.size() == 0 && PyErr_Occurred()){
                Py_XDECREF(self);
                return -1;
            }
			*/
            if (type == NULL){
                type = _default_type;
            }
            self->id_ = create_Id_from_path(path, numData, isGlobal, type);
            if (self->id_ == Id() && PyErr_Occurred()){
                Py_XDECREF(self);
                return -1;
            }
            return 0;
        }
        // The arguments could not be parsed as (path, dims, class),
        // try to parse it as an existing Id
        PyErr_Clear();        
        if (PyArg_ParseTuple(args, "O:moose_Id_init", &src) && Id_SubtypeCheck(src)){
            self->id_ = ((_Id*)src)->id_;
            return 0;
        }
        // The arguments could not be parsed as (path, dims, class), or Id
        // try to parse it as an existing ObjId
        PyErr_Clear();        
        if (PyArg_ParseTuple(args, "O:moose_Id_init", &src) && ObjId_SubtypeCheck(src)){
            self->id_ = ((_ObjId*)src)->oid_.id;
            return 0;
        }
        // Next try to parse it as an integer value for an existing Id
        PyErr_Clear(); // clear the error from parsing error
        if (PyArg_ParseTuple(args, "I:moose_Id_init", &id)){
            self->id_ = Id(id);
            return 0;
        }
        Py_XDECREF(self);
        return -1;
    }// ! moose_Id_init

     long moose_Id_hash(_Id * self)
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
    PyObject * deleteId(_Id * obj)
    {
#ifndef NDEBUG
        if (verbosity > 1){
            cout << "Deleting Id " << obj->id_ << endl;
        }
#endif
        string class_name = Field<string >::get(obj->id_, "className");
        vector <string> destFields = getFieldNames(class_name, "destFinfo");
        vector <string> lookupFields = getFieldNames(class_name, "lookupFinfo");
        vector <string> elementFields = getFieldNames(class_name, "elementFinfo");
        unsigned int numData = Field<unsigned int>::get(obj->id_, "numData");
        // clean up the maps containing initialized lookup/dest/element fields
        for (unsigned int ii = 0; ii < numData; ++ii){
            ObjId el(obj->id_, ii);
#ifndef NDEBUG
            if (verbosity > 1){
                cout << "\tDeleting ObjId " << el << endl;
            }
#endif
            for (unsigned int fidx = 0; fidx < lookupFields.size(); ++fidx){
                map<string, PyObject *>::iterator it =
                        get_inited_lookupfields().find(el.path() + "." + lookupFields[fidx]);
                if( it != get_inited_lookupfields().end()){
                    Py_XDECREF(it->second);
                    get_inited_lookupfields().erase(it);                    
                }
            }
            for (unsigned int fidx = 0; fidx < destFields.size(); ++fidx){
                map<string, PyObject *>::iterator it =
                        get_inited_destfields().find(el.path() + "." + destFields[fidx]);
                if( it != get_inited_destfields().end()){
                    Py_XDECREF(it->second);
                    get_inited_destfields().erase(it);
                }
            }
            for (unsigned int fidx = 0; fidx < elementFields.size(); ++fidx){
                map<string, PyObject *>::iterator it =
                        get_inited_elementfields().find(el.path() + "." + elementFields[fidx]);
                if( it != get_inited_elementfields().end()){
                    Py_XDECREF(it->second);
                    get_inited_elementfields().erase(it);
                }
            }    
        }
        SHELLPTR->doDelete(obj->id_);
        obj->id_ = Id();
        Py_RETURN_NONE;
    }
    
    PyObject * moose_Id_delete(_Id * self)
    {
        if (self->id_ == Id()){
            PyErr_SetString(PyExc_ValueError, "Cannot delete moose shell.");
            return NULL;
        }
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(NULL, "moose_Id_delete");
        }
        deleteId(self);
        Py_CLEAR(self);
        Py_RETURN_NONE;
    }
    
     PyObject * moose_Id_repr(_Id * self)
    {
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(NULL, "moose_Id_repr");
        }
        ostringstream repr;
        repr << "<moose.vec: class="
             << Field<string>::get(self->id_, "className") << ", "
             << "id=" << self->id_.value() << ","
             << "path=" << self->id_.path() << ">";
        return PyString_FromString(repr.str().c_str());
    } // !  moose_Id_repr

    // The string representation is unused. repr is used everywhere.
     PyObject * moose_Id_str(_Id * self)
    {
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(NULL, "moose_Id_str");
        }        
        return PyString_FromFormat("<moose.vec: class=%s, id=%u, path=%s>",
                                   Field<string>::get(self->id_, "className").c_str(),
                                   self->id_.value(), self->id_.path().c_str());
    } // !  moose_Id_str

    // 2011-03-23 15:09:19 (+0530)
     PyObject* moose_Id_getValue(_Id * self)
    {
        unsigned int id = self->id_.value();        
        PyObject * ret = Py_BuildValue("I", id);
        return ret;
    }
    
    /**
       Not to be redone. 2011-03-23 14:42:48 (+0530)
    */
     PyObject * moose_Id_getPath(_Id * self)
    {
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(NULL, "moose_Id_getPath");
        }        
        string path = self->id_.path();
        string default_end("[0]");
        if (endswith(path, default_end)){
            path.erase(path.length() - default_end.length(), default_end.length());
        }
        PyObject * ret = Py_BuildValue("s", path.c_str());
        return ret;
    }
    
    ////////////////////////////////////////////
    // Subset of sequence protocol functions
    ////////////////////////////////////////////
     Py_ssize_t moose_Id_getLength(_Id * self)
    {
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(-1, "moose_Id_getLength");
        }
        if (self->id_.element()->hasFields()){
            return (Py_ssize_t)(Field < unsigned int >::get(self->id_, "numField"));
        } else {
            return (Py_ssize_t)(self->id_.element()->numData());
        }
    }
    
    PyObject * moose_Id_getShape(_Id * self)
    {
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(NULL, "moose_Id_getShape");
        }       
        unsigned int numData = 1;
        if (self->id_.element()->hasFields()){
            numData = Field < unsigned int >::get(self->id_, "numField");
        } else {
            //            numData = Field < unsigned int >::get(self->id_, "numData");
            numData = self->id_.element()->numData();
        }
        PyObject * ret = PyTuple_New((Py_ssize_t)1);        
        if (PyTuple_SetItem(ret, (Py_ssize_t)0, Py_BuildValue("I", numData))){
                Py_XDECREF(ret);
                PyErr_SetString(PyExc_RuntimeError, "moose_Id_getShape: could not set tuple entry.");
                return NULL;
        }
        return ret;
    }
    
    PyObject * moose_Id_getItem(_Id * self, Py_ssize_t index)
    {
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(NULL, "moose_Id_getItem");
        }        
        extern PyTypeObject ObjIdType;
        if (index < 0){
            index += moose_Id_getLength(self);
        }
        if ((index < 0) || (index >= moose_Id_getLength(self))){
            PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
            return NULL;
        }
        _ObjId * ret = PyObject_New(_ObjId, &ObjIdType);
        ObjId oid(self->id_.path()); // This is just to get the dataIndex of parent
        if (self->id_.element()->hasFields()){
            // How to efficiently get the dataIndex of parent element
            // without creating ObjId from path?
            ret->oid_ = ObjId(self->id_, oid.dataIndex, index);
        } else {
            ret->oid_ = ObjId(self->id_, index, 0);
        }
        return (PyObject*)ret;
    }
    
    PyObject * moose_Id_getSlice(_Id * self, Py_ssize_t start, Py_ssize_t end)
    {
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(NULL, "moose_Id_getSlice");
        }        
        extern PyTypeObject ObjIdType;
        Py_ssize_t len = moose_Id_getLength(self);
        while (start < 0){
            start += len;
        }
        while (end < 0){
            end += len;
        }
        if (start > end){
            // PyErr_SetString(PyExc_IndexError, "Start index must be less than end.");
            // return NULL;
            // Python itself returns empty tuple in such cases, follow that
            return PyTuple_New(0);
        }
        PyObject * ret = PyTuple_New((Py_ssize_t)(end - start));
        ObjId oid(self->id_.path());
        // Py_XINCREF(ret);
        if (self->id_.element()->hasFields()){ // FieldElement - fieldIndex changing index
            for (unsigned int ii = start; ii < end; ++ii){
                _ObjId * value = PyObject_New(_ObjId, &ObjIdType);
                value->oid_ = ObjId(self->id_, oid.dataIndex, ii);
                if (PyTuple_SetItem(ret, (Py_ssize_t)(ii-start), (PyObject*)value)){
                    Py_XDECREF(ret);
                    Py_XDECREF(value);
                    PyErr_SetString(PyExc_RuntimeError, "Could assign tuple entry.");
                    return NULL;
                }
            }
        } else { // Ordinary element - dataIndex changing index
            for (unsigned int ii = start; ii < end; ++ii){
                _ObjId * value = PyObject_New(_ObjId, &ObjIdType);
                value->oid_ = ObjId(self->id_, ii);
                if (PyTuple_SetItem(ret, (Py_ssize_t)(ii-start), (PyObject*)value)){
                    Py_XDECREF(ret);
                    Py_XDECREF(value);
                    PyErr_SetString(PyExc_RuntimeError, "Could assign tuple entry.");
                    return NULL;
                }
            }
        }
        return ret;
    }

    ///////////////////////////////////////////////////
    // Mapping protocol
    ///////////////////////////////////////////////////
     PyObject * moose_Id_subscript(_Id * self, PyObject *op)
    {
        if (PyInt_Check(op) || PyLong_Check(op)){
            Py_ssize_t value = PyInt_AsLong(op);
            return moose_Id_getItem(self, value);
        } else {
            PyErr_SetString(PyExc_KeyError, "invalid index.");
            return NULL;
        }
    }
    
     PyObject * moose_Id_richCompare(_Id * self, PyObject * other, int op)
    {
        extern PyTypeObject IdType;
        bool ret = false;
        Id other_id = ((_Id*)other)->id_;
        if (!self || !other){
            ret = false;
        } else if (!PyObject_IsInstance(other, (PyObject*)&IdType)){
            ret = false;
        } else if (op == Py_EQ){
            ret = self->id_ == other_id;
        } else if (op == Py_NE) {
            ret = self->id_ != other_id;
        } else if (op == Py_LT){
            ret = self->id_ < other_id;
        } else if (op == Py_GT) {
            ret = other_id < self->id_;
        } else if (op == Py_LE){
            ret = (self->id_ < other_id) || (self->id_ == other_id);
        } else if (op == Py_GE){
            ret = (other_id < self->id_) || (self->id_ == other_id);
        }
        if (ret){
            Py_RETURN_TRUE;
        }
        Py_RETURN_FALSE;
    }
    
     int moose_Id_contains(_Id * self, PyObject * obj)
    {
        extern PyTypeObject ObjIdType;
        int ret = 0;
        if (ObjId_SubtypeCheck(obj)){
            ret = (((_ObjId*)obj)->oid_.id == self->id_);
        }
        return ret;
    }
    
     PyObject * moose_Id_getattro(_Id * self, PyObject * attr)
    {
        extern PyTypeObject ObjIdType;
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(NULL, "moose_Id_getattro");
        }        
        char * field = PyString_AsString(attr);
        PyObject * _ret = get_Id_attr(self, field);
        if (_ret != NULL){
            return _ret;
        }
        string class_name = Field<string>::get(self->id_, "className");
        string type = getFieldType(class_name, string(field), "valueFinfo");
        if (type.empty()){
            // Check if this field name is aliased and update fieldname and type if so.
            map<string, string>::const_iterator it = get_field_alias().find(string(field));
            if (it != get_field_alias().end()){
                field = const_cast<char*>((it->second).c_str());
                type = getFieldType(Field<string>::get(self->id_, "className"), it->second, "valueFinfo");
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

        switch (ftype){
            case 'd': {
                vector < double > val;
                Field< double >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 's': {
                vector < string > val;
                Field< string >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 'l': {
                vector < long > val;
                Field< long >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 'x': {
                vector < Id > val;
                Field< Id >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 'y': {
                vector < ObjId > val;
                Field< ObjId >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 'i': {
                vector < int > val;
                Field< int >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 'I': {
                vector < unsigned int > val;
                Field< unsigned int >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 'k': {
                vector < unsigned long > val;
                Field< unsigned long >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 'f': {
                vector < float > val;
                Field< float >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }            
            case 'b': {                                                               
                vector<bool> val;
                Field< bool >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 'c': {
                vector < char > val;
                Field< char >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 'h': {
                vector < short > val;
                Field< short >::getVec(self->id_, string(field), val);
                return to_pytuple(&val, ftype);
            }
            case 'z': {
                PyErr_SetString(PyExc_NotImplementedError, "DataId handling not implemented yet.");
                return NULL;
            }
            default:
                PyErr_SetString(PyExc_ValueError, "unhandled field type.");
                return NULL;                
        }
    }
    
     PyObject * moose_Id_setField(_Id * self, PyObject * args)
    {
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(NULL, "moose_Id_setField");
        }        
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

     int moose_Id_setattro(_Id * self, PyObject * attr, PyObject *value)
    {
        if (!Id::isValid(self->id_)){
            RAISE_INVALID_ID(-1, "moose_Id_setattro");
        }
        char * fieldname = NULL;
        int ret = -1;
        if (PyString_Check(attr)){
            fieldname = PyString_AsString(attr);
        } else {
            PyErr_SetString(PyExc_TypeError, "Attribute name must be a string");
            return -1;
        }
        string moose_class = Field<string>::get(self->id_, "className");
        string fieldtype = getFieldType(moose_class, string(fieldname), "valueFinfo");
        if (fieldtype.length() == 0){
            // If it is instance of a MOOSE Id then throw
            // error (to avoid silently creating new attributes due to
            // typos). Otherwise, it must have been subclassed in
            // Python. Then we allow normal Pythonic behaviour and
            // consider such mistakes user's responsibility.
            string class_name = ((PyTypeObject*)PyObject_Type((PyObject*)self))->tp_name;
            if (class_name != "vec"){
                Py_INCREF(attr);
                ret = PyObject_GenericSetAttr((PyObject*)self, attr, value);
                Py_DECREF(attr);
                return ret;
            }
            ostringstream msg;
            msg << "'" << moose_class << "' class has no field '" << fieldname << "'" << endl;
            PyErr_SetString(PyExc_AttributeError, msg.str().c_str());
            return -1;
        }
        char ftype = shortType(fieldtype);
        Py_ssize_t length = moose_Id_getLength(self);
        bool is_seq = true;
        if (!PySequence_Check(value)){
            is_seq = false;
        } else if (length != PySequence_Length(value)){
            PyErr_SetString(PyExc_IndexError, "Length of the sequence on the right hand side does not match Id size.");
            return -1;
        }
        switch(ftype){
            case 'd': {//SET_VECFIELD(double, d)
                vector<double> _value;
                if (is_seq){
                    for (unsigned int ii = 0; ii < length; ++ii){
                        double v = PyFloat_AsDouble(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                } else {
                    double v = PyFloat_AsDouble(value);
                    _value.assign(length, v);
                }
                ret = Field<double>::setVec(self->id_, string(fieldname), _value);
                break;
            }                
            case 's': {
                vector<string> _value;
                if (is_seq){
                    for (unsigned int ii = 0; ii < length; ++ii){
                        char * v = PyString_AsString(PySequence_GetItem(value, ii));
                        _value.push_back(string(v));
                    }
                } else {
                    char * v = PyString_AsString(value);
                    _value.assign(length, string(v));
                }                    
                ret = Field<string>::setVec(self->id_, string(fieldname), _value);
                break;
            }
            case 'i': {
                vector<int> _value;
                if (is_seq){
                    for (unsigned int ii = 0; ii < length; ++ii){
                        int v = PyInt_AsLong(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                } else {
                    int v = PyInt_AsLong(value);
                    _value.assign(length, v);
                }
                ret = Field< int >::setVec(self->id_, string(fieldname), _value);
                break;
            }
            case 'I': {//SET_VECFIELD(unsigned int, I)
                vector<unsigned int> _value;
                if (is_seq){
                    for (unsigned int ii = 0; ii < length; ++ii){
                        unsigned int v = PyInt_AsUnsignedLongMask(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                } else {
                    unsigned int v = PyInt_AsUnsignedLongMask(value);
                    _value.assign(length, v);
                }
                ret = Field< unsigned int >::setVec(self->id_, string(fieldname), _value);                
                break;
            }
            case 'l': {//SET_VECFIELD(long, l)
                vector<long> _value;
                if (is_seq){
                    for (unsigned int ii = 0; ii < length; ++ii){
                        long v = PyInt_AsLong(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                } else {
                    long v = PyInt_AsLong(value);
                    _value.assign(length, v);                    
                }
                ret = Field<long>::setVec(self->id_, string(fieldname), _value);
                break;
            }
            case 'k': {//SET_VECFIELD(unsigned long, k)
                vector<unsigned long> _value;
                if (is_seq){
                    for (unsigned int ii = 0; ii < length; ++ii){
                        unsigned long v = PyInt_AsUnsignedLongMask(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                } else {
                    unsigned long v = PyInt_AsUnsignedLongMask(value);
                    _value.assign(length, v);
                }
                ret = Field< unsigned long >::setVec(self->id_, string(fieldname), _value);                
                break;
            }
            case 'b': {
                vector<bool> _value;
                if (is_seq){
                    for (unsigned int ii = 0; ii < length; ++ii){
                        PyObject * _v = PySequence_GetItem(value, ii);
                        bool v = (Py_True ==_v) || (PyInt_AsLong(_v) != 0);
                        _value.push_back(v);
                    }
                } else {
                    bool v = (Py_True ==value) || (PyInt_AsLong(value) != 0);
                    _value.assign(length, v);
                }
                ret = Field< bool >::setVec(self->id_, string(fieldname), _value);
                break;
            }
            case 'c': {
                vector<char> _value;
                if (is_seq){
                    for (unsigned int ii = 0; ii < length; ++ii){
                        PyObject * _v = PySequence_GetItem(value, ii);
                        char * v = PyString_AsString(_v);
                        if (v && v[0]){
                            _value.push_back(v[0]);
                        } else {
                            ostringstream err;
                            err << ii << "-th element is NUL";
                            PyErr_SetString(PyExc_ValueError, err.str().c_str());
                            return -1;
                        }
                    }
                } else {
                    char * v = PyString_AsString(value);
                    if (v && v[0]){
                        _value.assign(length, v[0]);
                    } else {
                        PyErr_SetString(PyExc_ValueError,  "value is an empty string");
                        return -1;
                    }
                }                    
                ret = Field< char >::setVec(self->id_, string(fieldname), _value);
                break;
            }
            case 'h': {
                vector<short> _value;
                if (is_seq){
                    for (unsigned int ii = 0; ii < length; ++ii){
                        short v = PyInt_AsLong(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                } else {
                    short v = PyInt_AsLong(value);
                    _value.assign(length, v);
                }
                ret = Field< short >::setVec(self->id_, string(fieldname), _value);
                break;
            }
            case 'f': {//SET_VECFIELD(float, f)
                vector<float> _value;
                if (is_seq){
                    for (unsigned int ii = 0; ii < length; ++ii){
                        float v = PyFloat_AsDouble(PySequence_GetItem(value, ii));
                        _value.push_back(v);
                    }
                } else {
                    float v = PyFloat_AsDouble(value);                    
                    _value.assign(length, v);
                }
                ret = Field<float>::setVec(self->id_, string(fieldname), _value);
                break;
            }
            default:                
                break;
        }
        // MOOSE Field::set returns 1 for success 0 for
        // failure. Python treats return value 0 from setters as
        // success, anything else failure.
        if (ret || (PyErr_Occurred() == NULL)){
            return 0;
        } else {
            return -1;
        }
        
    }
    

} // end extern "C"

// 
// vec.cpp ends here
