// melement.cpp --- 
// 
// Filename: melement.cpp
// Description: 
// Author: 
// Maintainer: 
// Created: Mon Jul 22 16:50:41 2013 (+0530)
// Version: 
// Last-Updated: Tue Jul 23 20:35:28 2013 (+0530)
//           By: subha
//     Update #: 75
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// Mon Jul 22 16:50:47 IST 2013 - Taking out ObjId stuff from
// moosemodule.cpp
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

extern "C" {

    extern PyTypeObject ObjIdType;
    extern PyTypeObject IdType;


    
    PyObject * get_ObjId_attr(_ObjId * oid, string attribute)
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

    int moose_ObjId_init_from_id(PyObject * self, PyObject * args, PyObject * kwargs)
    {
        extern PyTypeObject ObjIdType;
        // The char arrays are to avoid deprecation warning
        char _id[] = "id";
        char _dataIndex[] = "dataIndex";
        char _fieldIndex[] = "fieldIndex";
        char _numFieldBits[] = "numFieldBits";
        static char * kwlist[] = {_id, _dataIndex, _fieldIndex, _numFieldBits, NULL};
        _ObjId * instance = (_ObjId*)self;
        unsigned int id = 0, data = 0, field = 0, numFieldBits = 0;
        PyObject * obj = NULL;
        if ((kwargs && PyArg_ParseTupleAndKeywords(args, kwargs,
                                                   "I|III:moose_ObjId_init",
                                                   kwlist,
                                                   &id, &data, &field, &numFieldBits))
            || (!kwargs && PyArg_ParseTuple(args, "I|III:moose_ObjId_init_from_id",
                                            &id, &data, &field, &numFieldBits))){
            PyErr_Clear();
            if (!Id::isValid(id)){
                RAISE_INVALID_ID(-1, "moose_ObjId_init_from_id");
            }
            instance->oid_ = ObjId(Id(id), DataId(data, field, numFieldBits));
            return 0;
        }
        PyErr_Clear();
        if ((kwargs && PyArg_ParseTupleAndKeywords(args,
                                                   kwargs,
                                                   "O|III:moose_ObjId_init_from_id",
                                                   kwlist,
                                                   &obj,
                                                   &data,
                                                   &field,
                                                   &numFieldBits)) ||
            (!kwargs && PyArg_ParseTuple(args,
                                         "O|III:moose_ObjId_init_from_id",
                                         &obj,
                                         &data,
                                         &field,
                                         &numFieldBits))){
            PyErr_Clear();
            // If first argument is an Id object, construct an ObjId out of it
            if (Id_Check(obj)){
                if (!Id::isValid(((_Id*)obj)->id_)){
                    RAISE_INVALID_ID(-1, "moose_ObjId_init_from_id");
                }                    
                instance->oid_ = ObjId(((_Id*)obj)->id_,
                                       DataId(data, field, numFieldBits));
                return 0;
            } else if (PyObject_IsInstance(obj, (PyObject*)&ObjIdType)){
                if (!Id::isValid(((_ObjId*)obj)->oid_.id)){
                    RAISE_INVALID_ID(-1, "moose_ObjId_init_from_id");
                }                    
                instance->oid_ = ((_ObjId*)obj)->oid_;
                return 0;
            }
        }
        return -1;
    }

    int moose_ObjId_init_from_path(PyObject * self, PyObject * args,
                                   PyObject * kwargs)
    {
        PyObject * dims = NULL;
        char * path = NULL;
        char * type = NULL;
        char _path[] = "path";
        char _dtype[] = "dtype";
        char _dims[] = "dims";
        static char * kwlist [] = {_path, _dims, _dtype, NULL};
        _ObjId * instance = (_ObjId*)self;
        instance->oid_ = ObjId::bad();

        // First try to parse the arguments as (path, dims, class)
        bool parse_success = false;
        if (kwargs == NULL){
            if (PyArg_ParseTuple(args,
                                 "s|Os:moose_ObjId_init_from_path",
                                 &path,
                                 &dims,
                                 &type)){
                parse_success = true;
            }
        } else if (PyArg_ParseTupleAndKeywords(args,
                                               kwargs,
                                               "s|Os:moose_ObjId_init_from_path",
                                               kwlist,
                                               &path,
                                               &dims,
                                               &type)){\
            parse_success = true;
        }
        PyErr_Clear();
        if (!parse_success){
            return -2;
        }    
        // First see if there is an existing object with at path
        instance->oid_ = ObjId(path);
        if (!(ObjId::bad() == instance->oid_)){
            return 0;
        }
        string basetype_str;
        if (type == NULL){
            basetype_str = get_baseclass_name(self);
        } else {
            basetype_str = string(type);
        }
        if (basetype_str.length() == 0){
            PyErr_SetString(PyExc_TypeError, "Unknown class. Need a valid MOOSE class or subclass thereof.");
            // Py_XDECREF(self);
            return -1;
        }
        
        Id new_id = create_Id_from_path(path, pysequence_to_dimvec(dims), basetype_str);
        if (new_id == Id() && PyErr_Occurred()){
            // Py_XDECREF(self);
            return -1;
        }
        instance->oid_ = ObjId(new_id);
        return 0;
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
                 "id : ematrix or integer\n"
                 "id of an existing element.\n"
                 "\n");
        
    int moose_ObjId_init(PyObject * self, PyObject * args,
                         PyObject * kwargs)
    {
        if (self && !PyObject_IsInstance(self, (PyObject*)Py_TYPE(self))){
            ostringstream error;
            error << "Expected an melement or subclass. Found "
                  << Py_TYPE(self)->tp_name;
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            return -1;
        }
        int ret = moose_ObjId_init_from_path(self, args, kwargs);
        if (ret >= -1){
            return ret;
        }
        // parsing arguments as (path, dims, classname) failed. See if it is existing Id or ObjId.
        if (moose_ObjId_init_from_id(self, args, kwargs) == 0){
            return 0;
        }
        PyErr_SetString(PyExc_ValueError,
                        "Could not parse arguments. "
                        " Call __init__(path, dims, dtype, parent) or"
                        " __init__(id, dataIndex, fieldIndex, numFieldBits)");        
        return -1;
    }

    /**
       This function simple returns the python hash of the unique path
       of this object.
    */
    long moose_ObjId_hash(_ObjId * self)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(-1, "moose_ObjId_hash");
        }
        long ret = (long)(self->oid_.id.value());
        ret |= (O32_HOST_ORDER == O32_BIG_ENDIAN)? \
                ((long)(self->oid_.dataId.value() >> 32))    \
                :((long)(self->oid_.dataId.value() << 32));
        return ret;
    }
    
    PyObject * moose_ObjId_repr(_ObjId * self)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_repr");
        }
        ostringstream repr;
        repr << "<moose." << Field<string>::get(self->oid_, "class") << ": "
             << "id=" << self->oid_.id.value() << ", "
             << "dataId=" << self->oid_.dataId.value() << ", "
             << "path=" << self->oid_.path() << ">";
        return PyString_FromString(repr.str().c_str());
    } // !  moose_ObjId_repr

    PyDoc_STRVAR(moose_ObjId_getId_documentation,
                 "getId()\n"
                 "\n"
                 "Get the ematrix of this object\n"
                 "\n");
    PyObject* moose_ObjId_getId(_ObjId * self)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_getId");
        }
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
                 "----------\n"
                 "fieldName : string\n"
                 "\tName of the field to be queried.\n"
                 "finfoType : string\n"
                 "\tFinfotype the field should be looked in for (can be \n"
                 "valueFinfo, srcFinfo, destFinfo, lookupFinfo)\n"
                 "\n");
    
    PyObject * moose_ObjId_getFieldType(_ObjId * self, PyObject * args)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_getFieldType");
        }
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
                 "----------\n"
                 "fieldName : string\n"
                 "\tName of the field.");
    PyObject * moose_ObjId_getField(_ObjId * self, PyObject * args)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_getField");
        }
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
    PyObject * moose_ObjId_getattro(_ObjId * self, PyObject * attr)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_getattro");
        }
        extern PyTypeObject IdType;
        extern PyTypeObject ObjIdType;
        const char * field;
        char ftype;
        if (PyString_Check(attr)){
            field = PyString_AsString(attr);
        } else {
            return PyObject_GenericGetAttr((PyObject*)self, attr);
        }
        PyObject * _ret = get_ObjId_attr(self, field);
        if (_ret != NULL){
            return _ret;
        }

	if (self->oid_ == ObjId::bad()){
            PyErr_SetString(PyExc_RuntimeError, "bad ObjId.");
            return NULL;
	}
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
        string fieldname(field);
        switch(ftype){
            case 's': {
                string _s = Field<string>::get(self->oid_, fieldname);
                return Py_BuildValue("s", _s.c_str());
            }
            case 'd': {
                double value = Field< double >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'i': {
                int value = Field<int>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'I': {
                unsigned int value = Field<unsigned int>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'l': {
                long value = Field<long>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'L': {
                long long value = Field<long long>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'k': {
                unsigned long value = Field<unsigned long>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'K': {
                unsigned long long value = Field<unsigned long long>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'f': {
                float value = Field<float>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'x': {                    
                Id value = Field<Id>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'y': {                    
                ObjId value = Field<ObjId>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'z': {
                PyErr_SetString(PyExc_NotImplementedError, "DataId handling not implemented yet.");
                return NULL;
            }
            case 'D': {
                vector< double > value = Field< vector < double > >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'X': { // vector<Id>
                vector < Id > value = Field<vector <Id> >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            } 
            case 'Y': { // vector<ObjId>
                vector < ObjId > value = Field<vector <ObjId> >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            } 
            case 'M': {
                vector< long > value = Field< vector <long> >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'P': {
                vector < unsigned long > value = Field< vector < unsigned long > >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'S': {
                vector < string > value = Field<vector <string> >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'v': {
                vector < int > value = Field<vector <int> >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'N': {
                vector <unsigned int > value = Field< vector < unsigned int> >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'T': { // vector<vector < unsigned int >>
                vector < vector < unsigned int > > value = Field<vector <vector < unsigned int > > >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            } 
            case 'Q': { // vector< vector < int > >
                vector <  vector < int >  > value = Field<vector < vector < int > > >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            } 
            case 'R': { // vector< vector < double > >
                vector <  vector < double >  > value = Field<vector < vector < double > > >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'F': {
                vector <float> value = Field< vector < float > >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'c': {
                char value = Field<char>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'h': {
                short value = Field<short>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'H': {
                unsigned short value = Field<unsigned short>::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'w': {
                vector < short > value = Field<vector <short> >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }
            case 'C': {
                vector < char > value = Field<vector <char> >::get(self->oid_, fieldname);
                return to_py(&value, ftype);
            }

            case 'b': {
                bool value = Field<bool>::get(self->oid_, fieldname);
                if (value){
                    Py_RETURN_TRUE;
                } else {
                    Py_RETURN_FALSE;
                }
            }

            default:
                return PyObject_GenericGetAttr((PyObject*)self, attr);
        }
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
                 "----------\n"
                 "fieldName : string\n"
                 "\tField to be assigned value to.\n"
                 "value : python datatype compatible with the type of the field\n"
                 "\tThe value to be assigned to the field.");
    
    PyObject * moose_ObjId_setField(_ObjId * self, PyObject * args)
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
    int  moose_ObjId_setattro(_ObjId * self, PyObject * attr, PyObject * value)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(-1, "moose_ObjId_setattro");
        }
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
            case 'd': {
                double _value = PyFloat_AsDouble(value);
                ret = Field<double>::set(self->oid_, string(field), _value);
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
                    PyErr_SetString(PyExc_ValueError, "Null pointer passed as ematrix Id value.");
                    return -1;
                }
                break;
            }
            case 'y': {// ObjId
                if (value){
                    ret = Field<ObjId>::set(self->oid_, string(field), ((_ObjId*)value)->oid_);
                } else {
                    PyErr_SetString(PyExc_ValueError, "Null pointer passed as ematrix Id value.");
                    return -1;
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
            case 'N': {//SET_VECFIELD(unsigned int, I)
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
            case 'T': {// vector< vector<unsigned int> >
                vector < vector <unsigned> > * _value = (vector < vector <unsigned> > *)to_cpp(value, ftype);
                if (!PyErr_Occurred()){
                    ret = Field < vector < vector <unsigned> > >::set(self->oid_, string(field), *_value);
                }
                delete _value;
                break;
            }
            case 'Q': {// vector< vector<int> >
                vector < vector <int> > * _value = (vector < vector <int> > *)to_cpp(value, ftype);
                if (!PyErr_Occurred()){
                    ret = Field < vector < vector <int> > >::set(self->oid_, string(field), *_value);
                }
                delete _value;
                break;
            }
            case 'R': {// vector< vector<double> >
                vector < vector <double> > * _value = (vector < vector <double> > *)to_cpp(value, ftype);
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
        switch(key_type_code){
            case 'b': {
                ret = lookup_value <bool> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'c': {
                ret = lookup_value <char> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'h': {
                ret = lookup_value <short> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }            
            case 'H': {
                ret = lookup_value <unsigned short> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }            
            case 'i': {
                ret = lookup_value <int> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }            
            case 'I': {
                ret = lookup_value <unsigned int> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }            
            case 'l': {
                ret = lookup_value <long> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }                        
            case 'k': {
                ret = lookup_value <unsigned long> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }                        
            case 'L': {
                ret = lookup_value <long long> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }                        
            case 'K': {
                ret = lookup_value <unsigned long long> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }                        
            case 'd': {
                ret = lookup_value <double> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }                        
            case 'f': {
                ret = lookup_value <float> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 's': {
                ret = lookup_value <string> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'x': {
                ret = lookup_value <Id> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'y': {
                ret = lookup_value <ObjId> (target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'D': {
                ret = lookup_value < vector <double> >(target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }                
            case 'S': {
                ret = lookup_value < vector <string> >(target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'X': {
                ret = lookup_value < vector <Id> >(target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'Y': {
                ret = lookup_value < vector <ObjId> >(target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'v': {
                ret = lookup_value < vector <int> >(target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'N': {
                ret = lookup_value < vector <unsigned int> >(target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'P': {
                ret = lookup_value < vector <unsigned long> >(target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }
            case 'F': {
                ret = lookup_value < vector <float> >(target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }                
            case 'w': {
                ret = lookup_value < vector <short> >(target, string(fieldName), value_type_code, key_type_code, key);
                break;
            }   
            case 'C': {
                ret = lookup_value < vector <char> >(target, string(fieldName), value_type_code, key_type_code, key);
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
                 "Lookup entry for `key` in `fieldName`\n"
                 "\n"
                 "Parameters\n"
                 "----------\n"
                 "fieldName : string\n"
                 "\tName of the lookupfield.\n"
                 "key : appropriate type for key of the lookupfield (as in the dict"
                 " getFieldDict).\n"
                 "\tKey for the look-up.");

    PyObject * moose_ObjId_getLookupField(_ObjId * self, PyObject * args)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_getLookupField");
        }
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
            case 'I': {
                ret = set_lookup_value <unsigned int> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }            
            case 'k': {
                ret = set_lookup_value <unsigned long> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }                        
            case 's': {
                ret = set_lookup_value <string> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }
            case 'i': {
                ret = set_lookup_value <int> (target, string(fieldName), value_type_code, key_type_code, key, value);
                break;
            }            
            case 'l': {
                ret = set_lookup_value <long> (target, string(fieldName), value_type_code, key_type_code, key, value);
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
    
    PyObject * moose_ObjId_setLookupField(_ObjId * self, PyObject * args)
    {
        if (!Id::isValid(self->oid_.id)){
            return NULL;
        }
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
                 "`element.fieldname(arg0, ...)`\n"
                 "\n"
                 "Parameters\n"
                 "----------\n"
                 "The number and type of paramateres depend on the destFinfo to be\n"
                 "set. Use moose.doc('{classname}.{fieldname}') to get builtin\n"
                 "documentation on the destFinfo `fieldname`\n"
                 );
    
    PyObject * moose_ObjId_setDestField(_ObjId * self, PyObject * args)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_setDestField");
        }
        return _setDestField(((_ObjId*)self)->oid_, args);        
    }

    
    PyObject * _setDestField(ObjId oid, PyObject *args)
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
            char vtypecode = shortType(argType[ii]);
            switch (vtypecode){                    
                case 'f': case 'd': {
                    double param = PyFloat_AsDouble(arg);
                    argstream << param << ",";
                }
                    break;
                case 's': {
                    char * param = PyString_AsString(arg);
                    argstream << string(param) << ",";
                }
                    break;
                case 'i': case 'l': {
                    long param = PyInt_AsLong(arg);
                    if (param == -1 && PyErr_Occurred()){
                        return NULL;
                    }
                    argstream << param << ",";
                }
                    break;
                case 'I': case 'k':{
                    unsigned long param =PyLong_AsUnsignedLong(arg);
                    if (PyErr_Occurred()){
                        return NULL;
                    }
                    argstream << param << ",";                            
                }
                    break;
                case 'x': {
                    Id param;
                    if (Id_SubtypeCheck(arg)){
                        _Id * id = (_Id*)(arg);
                        if (id == NULL){
                            error << "argument should be an ematrix or an melement";
                            PyErr_SetString(PyExc_TypeError, error.str().c_str());
                            return NULL;                                
                        }
                        param = id->id_;
                    } else if (ObjId_SubtypeCheck(arg)){
                        _ObjId * oid = (_ObjId*)(arg);
                        if (oid == NULL){
                            error << "argument should be an ematrix or an melement";
                            PyErr_SetString(PyExc_TypeError, error.str().c_str());
                            return NULL;                                
                        }
                        param = oid->oid_.id;
                    }
                    if ( SetGet1<Id>::set(oid, string(fieldName), param)){
                        return Py_True;
                    }
                    return Py_False;
                }
                case 'y': {
                    ObjId param;
                    if (Id_SubtypeCheck(arg)){
                        _Id * id = (_Id*)(arg);
                        if (id == NULL){
                            error << "argument should be an ematrix or an melement";
                            PyErr_SetString(PyExc_TypeError, error.str().c_str());
                            return NULL;                                
                        }
                        param = ObjId(id->id_);
                    } else if (ObjId_SubtypeCheck(arg)){
                        _ObjId * oid = (_ObjId*)(arg);
                        if (oid == NULL){
                            error << "argument should be an ematrix or an melement";
                            PyErr_SetString(PyExc_TypeError, error.str().c_str());
                            return NULL;                                
                        }
                        param = oid->oid_;
                    }
                    if ( SetGet1<ObjId>::set(oid, string(fieldName), param)){
                        return Py_True;
                    }
                    return Py_False;
                }
                case 'c': {
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
                    
                    ////////////////////////////////////////////////////
                    // We do NOT handle multiple vectors. Use the argument
                    // list as a single vector argument.
                    ////////////////////////////////////////////////////
                case 'v': {
                    return _set_vector_destFinfo<int>(oid, string(fieldName), ii, arg, vtypecode);
                }
                case 'w': {
                    return _set_vector_destFinfo<short>(oid, string(fieldName), ii, arg, vtypecode);
                }
                case 'L': {//SET_VECFIELD(long, l) {
                    return _set_vector_destFinfo<long>(oid, string(fieldName), ii, arg, vtypecode);
                }            
                case 'N': { //SET_VECFIELD(unsigned int, I)
                    return _set_vector_destFinfo<unsigned int>(oid, string(fieldName), ii, arg, vtypecode);
                }
                case 'K': {//SET_VECFIELD(unsigned long, k)
                    return _set_vector_destFinfo<unsigned long>(oid, string(fieldName), ii, arg, vtypecode);
                }
                case 'F': {//SET_VECFIELD(float, f)
                    return _set_vector_destFinfo<float>(oid, string(fieldName), ii, arg, vtypecode);
                }
                case 'D': {//SET_VECFIELD(double, d)
                    return _set_vector_destFinfo<double>(oid, string(fieldName), ii, arg, vtypecode);
                }                
                case 'S': {
                    return _set_vector_destFinfo<string>(oid, string(fieldName), ii, arg, vtypecode);
                }
                case 'X': {
                    return _set_vector_destFinfo<Id>(oid, string(fieldName), ii, arg, vtypecode);
                }
                case 'Y': {
                    return _set_vector_destFinfo<ObjId>(oid, string(fieldName), ii, arg, vtypecode);
                }
                default: {
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
            ostringstream err;
            err << fieldName << " takes arguments (";
            for (unsigned int ii = 0; ii < argType.size(); ++ii){
                error << argType[ii] << ",";
            }
            error << ")";                    
            PyErr_SetString(PyExc_TypeError, err.str().c_str());
            return NULL;
        }
    } // moose_ObjId_setDestField

    PyDoc_STRVAR(moose_ObjId_getFieldNames_documenation,
                 "getFieldNames(fieldType='')\n"
                 "\n"
                 "Get the names of fields on this element.\n"
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
                 "~~~~\n"
                 ">>> moose.getFieldNames('PulseGen', 'srcFinfo')\n"
                 "('childMsg', 'outputOut')\n"
                 "~~~~\n"
                 "\n");
    // 2011-03-23 15:28:26 (+0530)
    PyObject * moose_ObjId_getFieldNames(_ObjId * self, PyObject *args)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_getFieldNames");
        }
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
                 "Get the objects connected to this element by a message on specified\n"
                 "field.\n"
                 "\n"
                 "Parameters\n"
                 "----------\n"
                 "fieldName : str\n"
                 "\tname of the connection field (a destFinfo or srcFinfo)\n"
                 "\n"
                 "Returns\n"
                 "-------\n"
                 "out: tuple of ematrices.\n"
                 "\n");
                 
    PyObject * moose_ObjId_getNeighbors(_ObjId * self, PyObject * args)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_getNeighbors");
        }
        char * field = NULL;
        if (!PyArg_ParseTuple(args, "s:moose_ObjId_getNeighbors", &field)){
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
    PyDoc_STRVAR(moose_ObjId_connect_documentation,
                 "connect(srcfield, destobj, destfield, msgtype) -> bool\n"
                 "Connect another object via a message.\n"
                 "Parameters\n"
                 "----------\n"
                 "srcfield : str\n"
                 "\tsource field on self.\n"
                 "destobj : element\n"
                 "\tDestination object to connect to.\n"
                 "destfield : str\n"
                 "\tfield to connect to on `destobj`.\n"
                 "msgtype : str\n"
                 "\ttype of the message. Can be `Single`, `OneToAll`, `AllToOne`,\n"
                 " `OneToOne`, `Reduce`, `Sparse`. Default: `Single`."
                 "\n"
                 "Returns\n"
                 "-------\n"
                 "element of the created message.\n"
                 "\n"
                 "See also\n"
                 "--------\n"
                 "moose.connect\n"
                 "\n"
                 );
    PyObject * moose_ObjId_connect(_ObjId * self, PyObject * args)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_connect");
        }
        extern PyTypeObject ObjIdType;        
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
        MsgId mid = SHELLPTR->doAddMsg(msgType,
                                       self->oid_,
                                       string(srcField),
                                       dest->oid_,
                                       string(destField));
        if (mid == Msg::bad){
            PyErr_SetString(PyExc_NameError,
                            "connect failed: check field names and type compatibility.");
            return NULL;
        }
        const Msg* msg = Msg::getMsg(mid);
        Eref mer = msg->manager();
        _ObjId* msgMgrId = (_ObjId*)PyObject_New(_ObjId, &ObjIdType);        
        msgMgrId->oid_ = mer.objId();
        return (PyObject*)msgMgrId;
    }

    PyDoc_STRVAR(moose_ObjId_richcompare_documentation,
                 "Compare two element instances. This just does a string comparison of\n"
                 "the paths of the element instances. This function exists only to\n"
                 "facilitate certain operations requiring sorting/comparison, like\n"
                 "using elements for dict keys. Conceptually only equality comparison is\n"
                 "meaningful for elements.\n"); 
    PyObject* moose_ObjId_richcompare(_ObjId * self, PyObject * other, int op)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_richcompare");
        }
        extern PyTypeObject ObjIdType;
        if ((self != NULL && other == NULL) || (self == NULL && other != NULL)){
            if (op == Py_EQ){
                Py_RETURN_FALSE;
            } else if (op == Py_NE){
                Py_RETURN_TRUE;
            } else {
                PyErr_SetString(PyExc_TypeError, "Cannot compare NULL with non-NULL");
                return NULL;
            }
        }
        if (!PyObject_IsInstance(other, (PyObject*)&ObjIdType)){
            ostringstream error;
            error << "Cannot compare ObjId with "
                  << Py_TYPE(other)->tp_name;
            PyErr_SetString(PyExc_TypeError, error.str().c_str());
            return NULL;
        }
        if (!Id::isValid(((_ObjId*)other)->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_richcompare");
        }

        string l_path = self->oid_.path();
        string r_path = ((_ObjId*)other)->oid_.path();
        int result = l_path.compare(r_path);
        if (result == 0){
            if (op == Py_EQ || op == Py_LE || op == Py_GE){
                Py_RETURN_TRUE;
            }
            Py_RETURN_FALSE;
        } else if (result < 0){
            if (op == Py_LT || op == Py_LE || op == Py_NE){
                Py_RETURN_TRUE;
            }
            Py_RETURN_FALSE;
        } else {
            if (op == Py_GT || op == Py_GE || op == Py_NE){
                Py_RETURN_TRUE;
            }
            Py_RETURN_FALSE;
        }
    }

    PyDoc_STRVAR(moose_ObjId_getDataIndex_documentation,
                 "getDataIndex()\n"
                 "\n"
                 "Return the dataIndex of this object.\n");
    PyObject * moose_ObjId_getDataIndex(_ObjId * self)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_getDataIndex");
        }        
        PyObject * ret = Py_BuildValue("I", self->oid_.dataId.value());
        return ret;
    }

    // WARNING: fieldIndex has been deprecated in dh_branch. This
    // needs to be updated accordingly.  The current code is just
    // place-holer to avoid compilation errors.
    PyObject * moose_ObjId_getFieldIndex(_ObjId * self)
    {
        if (!Id::isValid(self->oid_.id)){
            RAISE_INVALID_ID(NULL, "moose_ObjId_getFieldIndex");
        }
        PyObject * ret = Py_BuildValue("I", self->oid_.dataId.value());
        return ret;
    }
    
    ///////////////////////////////////////////////
    // Python method lists for PyObject of ObjId
    ///////////////////////////////////////////////

    static PyMethodDef ObjIdMethods[] = {
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
        {"ematrix", (PyCFunction)moose_ObjId_getId, METH_NOARGS,
         "Return the ematrix this element belongs to."},
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
    PyDoc_STRVAR(moose_ObjId_documentation,
                 "Individual moose element contained in an array-type object\n"
                 "(ematrix). Each element has a unique path, possibly with its index in\n"
                 "the ematrix. These are identified by three components: id_ and\n"
                 "dindex. id_ is the Id of the containing ematrix, it has a unique\n"
                 "numerical value (field `value`). `dindex` is the index of the current\n"
                 "item in the containing ematrix. `dindex` is 0 for single elements.");
    PyTypeObject ObjIdType = { 
        PyVarObject_HEAD_INIT(NULL, 0)            /* tp_head */
        "moose.melement",                      /* tp_name */
        sizeof(_ObjId),                     /* tp_basicsize */
        0,                                  /* tp_itemsize */
        0,                                  /* tp_dealloc */
        0,                                  /* tp_print */
        0,                                  /* tp_getattr */
        0,                                  /* tp_setattr */
        0,       /* tp_compare */
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
        moose_ObjId_documentation,
        0,                                  /* tp_traverse */
        0,                                  /* tp_clear */
        (richcmpfunc)moose_ObjId_richcompare, /* tp_richcompare */
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

    

}// extern "C"

// 
// melement.cpp ends here
