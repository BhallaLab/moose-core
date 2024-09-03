/***
 *    Description:  Finfo Wrapper
 *
 *        Created:  2020-03-30

 *         Author:  Dilawar Singh <dilawar.s.rajput@gmail.com>
 *        License:  GPLv3
 */

#ifndef FINFO_H
#define FINFO_H

#include "../basecode/header.h"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace std;

class MooseVec;
// class ObjId;
// class Finfo;

///////////////////////////////////////////////////////////////
// Utility functions that do not need to be members
///////////////////////////////////////////////////////////////

// Get the ObjId for the FieldElement
ObjId getFieldObjId(const ObjId &oid, const Finfo *f);

template <typename T>
std::function<bool(T)> getSetGetFunc1(const ObjId &oid, const string &fname);

// Get ValueField
py::object getFieldValue(const ObjId &oid, const Finfo *f);

// DestFinfo setter function - depending on number of parameters switches between 1 and 2
py::cpp_function getDestFinfoSetterFunc(const ObjId &oid, const Finfo *finfo);
// Setter for single parameter DestFinfo
py::cpp_function getDestFinfoSetterFunc1(const ObjId &oid, const Finfo *finfo,
                                         const string &srctype);

// Get setter-function for two-parameter DestFinfo2, essentially SetGet2<type1, type2>::set(oid, fieldname, param1, param2)
py::cpp_function getDestFinfoSetterFunc2(const ObjId &oid, const Finfo *finfo,
                                         const string &srctype,
                                         const string &tgttype);
// Get ElementField
py::list getElementFinfo(const ObjId &objid, const Finfo *f);

// Get item from Element field
py::object getElementFinfoItem(const ObjId &oid, const Finfo *f, int i);

// Get number of elements in ElementField
unsigned int getNumField(const ObjId &oid, const Finfo *f);

// Set number of elements in ElementField
bool setNumField(const ObjId &oid, const Finfo *f, unsigned int i);

// Get item in LookupValueField: uses the inner function below
py::object getLookupValueFinfoItem(const ObjId &oid, const Finfo *f,
                                   const py::object &key);

// Utility function for modular code
template <typename T>
py::object getLookupValueFinfoItemInner(const ObjId &oid, const Finfo *f,
                                        const T &key, const string &tgtType);

// Set item in LookupValueField
bool setLookupValueFinfoItem(const ObjId &oid, const py::object &key,
                             const py::object &val, const Finfo *finfo);

// Get list of field names of given Finfo type ("src", "dest",
// "value", "lookup", "shared", and "fieldElement"). If finfoType =
// "*", then return all types
//
vector<pair<string, string>> finfoNames(const Cinfo *cinfo,
                                        const string &finfoType);


////////////////////////////////////////////////////////////
// Wrapper class for field elements
////////////////////////////////////////////////////////////

class __Finfo__ {
  public:
    __Finfo__(const ObjId &oid, const Finfo *f, const string &finfoType);

    // ObjId of the fieldElement
    ObjId getObjId() const;
  
    // Get attribute (python api);
    unsigned int getNumField();
    bool setNumField(unsigned int);
    
    // Exposed to python as __getitem__
    py::object getItem(const py::object &key);
    
    // Exposed to python as __setitem__
    bool setItem(const py::object &key, const py::object &val);
    
    py::object operator()(const py::object &key);
    
    string type() const;
    
    // Return a MooseVec element (copy).
    MooseVec getMooseVec();
    
    // Retun by pointer.
    MooseVec *getMooseVecPtr();
    
    function<py::object(const py::object &key)> func_;
  
  private:
    ObjId oid_;
    const Finfo *f_;
    const string finfoType_;
    // __Finfo__ needs to be copiable.
    shared_ptr<MooseVec> pVec_;
};

#endif /* end of include guard: FINFO_H */
