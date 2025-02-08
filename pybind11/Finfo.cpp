// =====================================================================================
//
//       Filename:  Finfo.cpp
//
//    Description:
//
//         Author:  Dilawar Singh (), dilawar.s.rajput@gmail.com
//   Organization:  NCBS Bangalore
//
// =====================================================================================

#include "Finfo.h"
#include "helper.h"
#include "pymoose.h"

#include "../basecode/header.h"
#include "../builtins/Variable.h"
#include "../utility/print_function.hpp"
#include "../utility/strutil.h"

ObjId getFieldObjId(const ObjId &oid, const Finfo *f)
{
    return ObjId(oid.path() + "/" + f->name());
}

template <typename T>
std::function<bool(T)> getSetGetFunc1(const ObjId &oid, const string &fname)
{
    std::function<bool(T)> func = [oid, fname](const T &val) {
        return SetGet1<T>::set(oid, fname, val);
    };
    // std::cout << "getSetGet1Func" << std::endl;
    return func;
}


py::object getFieldValue(const ObjId &oid, const Finfo *f)
{
    auto rttType = f->rttiType();
    auto fname = f->name();
    py::object r = py::none();
    if(rttType == "double" || rttType == "float") {
        r = py::float_(getField<double>(oid, fname));
    }
    else if(rttType == "vector<double>") {
        r = getFieldNumpy<double>(oid, fname);
    }
    else if(rttType == "vector<unsigned int>") {
        r = getFieldNumpy<unsigned int>(oid, fname);
    }
    else if(rttType == "string")
        r = py::str(getField<string>(oid, fname));
    else if(rttType == "char")
        r = py::int_(getField<char>(oid, fname));
    else if(rttType == "int")
        r = py::int_(getField<int>(oid, fname));
    else if(rttType == "unsigned int")
        r = py::int_(getField<unsigned int>(oid, fname));
    else if(rttType == "unsigned long")
        r = py::int_(getField<unsigned long>(oid, fname));
    else if(rttType == "bool")
        r = py::bool_(getField<bool>(oid, fname));
    else if(rttType == "Id")
        r = py::cast(getField<Id>(oid, fname));
    else if(rttType == "ObjId")
        r = py::cast(getField<ObjId>(oid, fname));
    else if(rttType == "Variable")
        r = py::cast(getField<Variable>(oid, fname));
    else if(rttType == "vector<Id>")
        r = py::cast(getField<vector<Id>>(oid, fname));
    else if(rttType == "vector<ObjId>")
        r = py::cast(getField<vector<ObjId>>(oid, fname));
    else if(rttType == "vector<string>")
        r = py::cast(getField<vector<string>>(oid, fname));
    else {
        MOOSE_WARN("Warning: getValueFinfo:: Unsupported type '" + rttType +
                   "'");
        r = py::none();
    }
    return r;
}

py::cpp_function getDestFinfoSetterFunc(const ObjId &oid, const Finfo *finfo)
{
    const auto rttType = finfo->rttiType();
    vector<string> types;
    moose::tokenize(rttType, ",", types);

    if(types.size() == 1)
        return getDestFinfoSetterFunc1(oid, finfo, types[0]);

    assert(types.size() == 2);
    return getDestFinfoSetterFunc2(oid, finfo, types[0], types[1]);
}

// Get DestFinfo1.
py::cpp_function getDestFinfoSetterFunc1(const ObjId &oid, const Finfo *finfo,
                                         const string &ftype)
{
    const auto fname = finfo->name();
    if(ftype == "void") {
        std::function<bool()> func = [oid, fname]() {
            return SetGet0::set(oid, fname);
        };
        return func;
    }

    if(ftype == "double")
        return getSetGetFunc1<double>(oid, fname);
    if(ftype == "ObjId")
        return getSetGetFunc1<ObjId>(oid, fname);
    if(ftype == "Id")
        return getSetGetFunc1<Id>(oid, fname);
    if(ftype == "string")
        return getSetGetFunc1<string>(oid, fname);
    if(ftype == "vector<Id>")
        return getSetGetFunc1<vector<Id>>(oid, fname);
    if(ftype == "vector<ObjId>")
        return getSetGetFunc1<vector<ObjId>>(oid, fname);
    if(ftype == "vector<double>")
        return getSetGetFunc1<vector<double>>(oid, fname);

    throw runtime_error("getFieldPropertyDestFinfo1::NotImplemented " + fname +
                        " for rttType " + ftype + " for oid " + oid.path());
}

// Get DestFinfo2
py::cpp_function getDestFinfoSetterFunc2(const ObjId &oid, const Finfo *finfo,
                                         const string &ftype1,
                                         const string &ftype2)
{
    const auto fname = finfo->name();
    if(ftype1 == "double") {
        if(ftype2 == "unsigned int") {
            std::function<bool(double, unsigned int)> func =
                [oid, fname](const double a, const unsigned int b) {
                    return SetGet2<double, unsigned int>::set(oid, fname, a, b);
                };
            return func;
        }
        if(ftype2 == "long") {
            std::function<bool(double, long)> func =
                [oid, fname](const double a, const long b) {
                    return SetGet2<double, long>::set(oid, fname, a, b);
                };
            return func;
        }
        if(ftype2 == "double") {
            std::function<bool(double, double)> func =
                [oid, fname](const double a, const double b) {
                    return SetGet2<double, double>::set(oid, fname, a, b);
                };
            return func;
        }
    }

    if(ftype1 == "string") {
        if(ftype2 == "string") {
            std::function<bool(string, string)> func = [oid, fname](string a,
                                                                    string b) {
                return SetGet2<string, string>::set(oid, fname, a, b);
            };
            return func;
        }
    }
    if(ftype1 == "ObjId" && ftype2 == "ObjId") {
        std::function<bool(ObjId, ObjId)> func = [oid, fname](ObjId a,
                                                              ObjId b) {
            return SetGet2<ObjId, ObjId>::set(oid, fname, a, b);
        };
        return func;
    }
    if(ftype1 == "vector<ObjId>" && ftype2 == "double") {
        std::function<bool(vector<ObjId>, double)> func =
            [oid, fname](vector<ObjId> a, double b) {
                return SetGet2<vector<ObjId>, double>::set(oid, fname, a, b);
            };
        return func;
    }

    throw runtime_error("getFieldPropertyDestFinfo2::NotImplemented " + fname +
                        " for rttType " + finfo->rttiType() + " for oid " +
                        oid.path());
}

py::list getElementFinfo(const ObjId &objid, const Finfo *f)
{
    auto fname = f->name();
    auto oid = ObjId(objid.path() + '/' + fname);
    auto len = Field<unsigned int>::get(oid, "numField");
    vector<ObjId> res(len);
    for(unsigned int i = 0; i < len; i++)
        res[i] = ObjId(oid.path(), oid.dataIndex, i);
    return py::cast(res);
}

py::object getElementFinfoItem(const ObjId &oid, const Finfo *f, int index)
{    
    size_t numFields = getNumField(oid, f);
    size_t i = (index < 0) ? (int)numFields + index : index;
    if(i >= numFields)
        throw py::index_error("Index " + to_string(i) + " out of range.");
    auto foid = getFieldObjId(oid, f);
    return py::cast(ObjId(foid.path(), foid.dataIndex, i));
}

unsigned int getNumField(const ObjId &oid, const Finfo *f)
{
    ObjId foid = getFieldObjId(oid, f);
    return Field<unsigned int>::get(foid, "numField");
}

bool setNumField(const ObjId &oid, const Finfo *f, unsigned int num)
{
    ObjId foid = getFieldObjId(oid, f);
    return Field<unsigned int>::set(foid, "numField", num);
}

py::object getLookupValueFinfoItem(const ObjId &oid, const Finfo *f,
                                   const py::object &key)
{

    auto rttType = f->rttiType();
    auto fname = f->name();
    vector<string> srcDestType;
    moose::tokenize(rttType, ",", srcDestType);
    string srcType = srcDestType[0];
    string tgtType = srcDestType[1];

    py::object r = py::none();

    if(srcType == "string")
        return getLookupValueFinfoItemInner<string>(oid, f, key.cast<string>(),
                                                    tgtType);
    if(srcType == "unsigned int")
        return getLookupValueFinfoItemInner<unsigned int>(
            oid, f, key.cast<unsigned int>(), tgtType);
    if(srcType == "ObjId")
        return getLookupValueFinfoItemInner<ObjId>(oid, f, key.cast<ObjId>(),
                                                   tgtType);
    if(srcType == "vector<double>")
        return getLookupValueFinfoItemInner<vector<double>>(
            oid, f, key.cast<vector<double>>(), tgtType);
    if(srcType == "Id")
        return getLookupValueFinfoItemInner<Id>(oid, f, key.cast<Id>(),
                                                tgtType);

    py::print("getLookupValueFinfoItem::NotImplemented for key:", key,
              "srcType:", srcType, "and tgtType:", tgtType,
              "path: ", oid.path());
    throw runtime_error("getLookupValueFinfoItem::NotImplemented error");
    return r;
}

template <typename T>
py::object getLookupValueFinfoItemInner(const ObjId &oid, const Finfo *f,
                                        const T &key, const string &tgtType)
{
    auto fname = f->name();
    if(tgtType == "bool")
        return py::cast(LookupField<T, bool>::get(oid, fname, key));
    if(tgtType == "double")
        return py::cast(LookupField<T, double>::get(oid, fname, key));
    if(tgtType == "unsigned int")
        return py::cast(LookupField<T, unsigned int>::get(oid, fname, key));
    if(tgtType == "int")
        return py::cast(LookupField<T, int>::get(oid, fname, key));
    if(tgtType == "string")
        return py::cast(LookupField<T, string>::get(oid, fname, key));
    if(tgtType == "ObjId")
        return py::cast(LookupField<T, ObjId>::get(oid, fname, key));
    if(tgtType == "Id")
        return py::cast(LookupField<T, Id>::get(oid, fname, key));
    if(tgtType == "vector<double>")
        return py::cast(LookupField<T, vector<double>>::get(oid, fname, key));
    if(tgtType == "vector<unsigned int>")
        return py::cast(
            LookupField<T, vector<unsigned int>>::get(oid, fname, key));
    if(tgtType == "vector<Id>")
        return py::cast(LookupField<T, vector<Id>>::get(oid, fname, key));
    if(tgtType == "vector<ObjId>")
        return py::cast(LookupField<T, vector<ObjId>>::get(oid, fname, key));
    if(tgtType == "vector<string>")
        return py::cast(LookupField<T, vector<string>>::get(oid, fname, key));

    py::print(__func__, ":: warning: Could not find", fname, "for key", key,
              "(type", tgtType, ") on path ", oid.path());
    throw py::key_error("Attribute error.");
    return py::none();
}


bool setLookupValueFinfoItem(const ObjId& oid, const py::object& key,
                                        const py::object& val,
                                        const Finfo* finfo)
{
    auto rttType = finfo->rttiType();
    auto fieldName = finfo->name();

    vector<string> srcDestType;
    moose::tokenize(rttType, ",", srcDestType);
    assert(srcDestType.size() == 2);

    auto srcType = srcDestType[0];
    auto destType = srcDestType[1];

    if(srcType == "unsigned int") {
        if(destType == "double")
            return LookupField<unsigned int, double>::set(
                oid, fieldName, py::cast<unsigned int>(key),
                py::cast<double>(val));
    }
    if(srcType == "string") {
        if(destType == "double")
            return LookupField<string, double>::set(
                oid, fieldName, py::cast<string>(key), py::cast<double>(val));
    }
    if(srcType == "string") {
        if(destType == "vector<double>")
            return LookupField<string, vector<double>>::set(
                oid, fieldName, py::cast<string>(key), py::cast<vector<double>>(val));
    }
    if(srcType == "string") {
        if(destType == "long")
            return LookupField<string, long>::set(
                oid, fieldName, py::cast<string>(key), py::cast<long>(val));
    }
    if(srcType == "string") {
        if(destType == "vector<long>")
            return LookupField<string, vector<long>>::set(
                oid, fieldName, py::cast<string>(key), py::cast<vector<long>>(val));
    }
    if(srcType == "string") {
        if(destType == "string")
            return LookupField<string, string>::set(
                oid, fieldName, py::cast<string>(key), py::cast<string>(val));
    }
    if(srcType == "string") {
        if(destType == "vector<string>")
            return LookupField<string, vector<string>>::set(
                oid, fieldName, py::cast<string>(key), py::cast<vector<string>>(val));
    }

    py::print("NotImplemented::setLookupValueFinfoItem:", key, "to value", val,
              "for object", oid.path(), "and fieldName=", fieldName,
              "rttiType=", rttType, srcDestType);
    throw runtime_error("NotImplemented");
    return true;
}

vector<pair<string, string>> finfoNames(const Cinfo *cinfo,
                                        const string &what = "*")
{

    vector<pair<string, string>> ret;

    if(!cinfo) {
        cerr << "Invalid class name." << endl;
        return ret;
    }

    if(what == "valueFinfo" || what == "value" || what == "*") {
        for(unsigned int ii = 0; ii < cinfo->getNumValueFinfo(); ++ii) {
            Finfo *finfo = cinfo->getValueFinfo(ii);
            ret.push_back({finfo->name(), finfo->rttiType()});
        }
    }
    else if(what == "srcFinfo" || what == "src" || what == "*") {
        for(unsigned int ii = 0; ii < cinfo->getNumSrcFinfo(); ++ii) {
            Finfo *finfo = cinfo->getSrcFinfo(ii);
            ret.push_back({finfo->name(), finfo->rttiType()});
        }
    }
    else if(what == "destFinfo" || what == "dest" || what == "*") {
        for(unsigned int ii = 0; ii < cinfo->getNumDestFinfo(); ++ii) {
            Finfo *finfo = cinfo->getDestFinfo(ii);
            ret.push_back({finfo->name(), finfo->rttiType()});
        }
    }
    else if(what == "lookupFinfo" || what == "lookup" || what == "*") {
        for(unsigned int ii = 0; ii < cinfo->getNumLookupFinfo(); ++ii) {
            Finfo *finfo = cinfo->getLookupFinfo(ii);
            ret.push_back({finfo->name(), finfo->rttiType()});
        }
    }
    else if(what == "sharedFinfo" || what == "shared" || what == "*") {
        for(unsigned int ii = 0; ii < cinfo->getNumSrcFinfo(); ++ii) {
            Finfo *finfo = cinfo->getSrcFinfo(ii);
            ret.push_back({finfo->name(), finfo->rttiType()});
        }
    }
    else if(what == "fieldElementFinfo" || what == "fieldElement" ||
            what == "*") {
        for(unsigned int ii = 0; ii < cinfo->getNumFieldElementFinfo(); ++ii) {
            Finfo *finfo = cinfo->getFieldElementFinfo(ii);
            ret.push_back({finfo->name(), finfo->rttiType()});
        }
    }
    return ret;
}

//////////////////////////////////////////////////////////////
// __Finfo__ class member functions
//////////////////////////////////////////////////////////////
__Finfo__::__Finfo__(const ObjId &oid, const Finfo *f, const string &finfoType)
    : oid_(oid), f_(f), finfoType_(finfoType), pVec_(nullptr)
{
     // why LookupFinfo for DestFinfo??? Pretty sure this was a mistake! - Subha
    // if(finfoType == "DestFinfo")
    //     func_ = [oid, f](const py::object &key) {
    // 	    cout << "Creating LookupFinfo for DestFinfo" << endl; // DEBUG
    //         return getLookupValueFinfoItem(oid, f, key);
    //     };
    // else
        if(finfoType == "FieldElementFinfo")
        func_ = [oid, f](const py::object &index) {
            // this is essential of make this function static.
            // Cast to int because we support negative indexing.
            return getElementFinfoItem(oid, f, py::cast<int>(index));
        };
    else if(finfoType == "LookupValueFinfo")
        func_ = [oid, f, this](const py::object &key) {
            // Assigning is essential or make these functions static.
            return getLookupValueFinfoItem(oid, f, key);
        };
    else
        func_ = [this](const py::object &key) {
            throw runtime_error("Not supported for Finfo type '" + finfoType_ +
                                "'");
            return py::none();
        };
}

ObjId __Finfo__::getObjId() const
{
    return getFieldObjId(oid_, f_);
}

unsigned int __Finfo__::getNumField()
{

    return Field<unsigned int>::get(oid_, "numField");
}

bool __Finfo__::setNumField(unsigned int num)
{
    return Field<unsigned int>::set(oid_, "numField", num);
}

py::object __Finfo__::getItem(const py::object &key)
{
    return this->func_(key);
}

// Exposed to python as __setitem__ on Finfo
bool __Finfo__::setItem(const py::object &key, const py::object &val)
{
    return setLookupValueFinfoItem(oid_, key, val, f_);
}

py::object __Finfo__::operator()(const py::object &key)
{
    return this->getItem(key);
}

// Return by copy.
MooseVec __Finfo__::getMooseVec()
{
    return MooseVec(this->getObjId());
}

MooseVec *__Finfo__::getMooseVecPtr()
{
    if(!pVec_)
        pVec_.reset(new MooseVec(this->getObjId()));
    return pVec_.get();
}

// py::object __Finfo__::getElementFinfoItem(int index)
// {
//     size_t numFields = this->getNumField();
//     size_t i = (index < 0) ? (int)numFields + index : index;
//     if(i >= numFields)
//         throw py::index_error("Index " + to_string(i) + " out of range.");
//     auto oid = this->getObjId()
//     return py::cast(ObjId(oid.path(), oid.dataIndex, i));
// }

string __Finfo__::type() const
{
    return finfoType_;
}

// Non static function.
