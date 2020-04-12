/***
 *    Description:  moose.vec class.
 *
 *        Created:  2020-03-30

 *         Author:  Dilawar Singh <dilawar.s.rajput@gmail.com>
 *        License:  GPLv3
 */

#ifndef MOOSE_VEC_H
#define MOOSE_VEC_H

#include "../external/pybind11/include/pybind11/pybind11.h"
#include "../external/pybind11/include/pybind11/numpy.h"
namespace py = pybind11;

class MooseVec
{

public:
    MooseVec(const string& path, unsigned int n, const string& dtype);

    MooseVec(const Id& id);

    MooseVec(const ObjId& oid);

    const ObjId& obj() const;

    const string dtype() const;

    const size_t size() const;

    vector<MooseVec> children() const;

    const string path() const;

    const string name() const;

    ObjId parent() const;

    unsigned int len();

    const ObjId& getItemRef(const int i) const;

    // Get vector element. Vector element could be `dataIndex` or `fieldIndex`.
    // Allows negative indexing.
    ObjId getItem(const int i) const;
    vector<ObjId> getItemRange(const py::slice& slice) const;

    ObjId getDataItem(const size_t i) const;
    ObjId getFieldItem(const size_t i) const;

    // Set attribute to vector.
    // void setAttrOneToAll(const string& name, const py::object& val);
    // void setAttrOneToOne(const string& name, const py::sequence& val);

    template <typename T>
    inline bool setAttributeAtIndex(size_t i, const string& name, T val,
                                    const string& rttType)
    {
        // Otherwise coerce the type. Be conservative here, please.
        if (rttType == "double")
            return Field<double>::set(getItem(i), name, val);
        if (rttType == "int")
            return Field<int>::set(getItem(i), name, val);
        if (rttType == "unsigned long")
            return Field<unsigned long>::set(getItem(i), name, val);
        if (rttType == "unsigned int")
            return Field<unsigned int>::set(getItem(i), name, val);
        return false;
    }

    template <typename T>
    bool setAttrOneToAll(const string& name, const T& val)
    {
        auto cinfo = oid_.element()->cinfo();
        auto finfo = cinfo->findFinfo(name);
        assert(finfo);

        string expectedType(finfo->rttiType());
        string givenType(Conv<T>::rttiType());

        bool isSameType = (expectedType == givenType);

        bool res = true;
        for (size_t i = 0; i < size(); i++)
        {
            if (isSameType)
            {
                res &= Field<T>::set(getItem(i), name, val);
                continue;
            }

            // else try coercing the types.
            if (!setAttributeAtIndex<T>(i, name, val, expectedType))
                throw py::value_error("Unexpected type '" + givenType +
                                      "', MOOSE could not convert to '" +
                                      expectedType + "'.");
        }
        return res;
    }

    template <typename T>
    bool setAttrOneToOne(const string& name, const vector<T>& val)
    {
        auto cinfo = oid_.element()->cinfo();
        auto finfo = cinfo->findFinfo(name);
        assert(finfo);

        string expectedType(finfo->rttiType());
        string recievedType(Conv<T>::rttiType());

        bool isSameType = expectedType == recievedType;

        if (val.size() != size())
            throw runtime_error(
                "Length of sequence on the right hand side "
                "does not match size of vector. "
                "Expected " +
                to_string(size()) + ", got " + to_string(val.size()));

        bool res = true;
        for (size_t i = 0; i < size(); i++)
        {
            if (isSameType)
            {
                res &= Field<T>::set(getItem(i), name, val[i]);
                continue;
            }

            // Else conservatively coerse value. Required for int -> double
            // etc.
            if (!setAttributeAtIndex<T>(i, name, val[i], expectedType))
                throw py::value_error("Unexpected type '" + recievedType +
                                      "', MOOSE could not convert to '" +
                                      expectedType + "'.");
        }

        return res;
    }

    // Get attributes.
    py::object getAttribute(const string& key);

    // Set attribute.
    bool setAttribute(const string& name, const py::object& val);

    vector<ObjId> objs() const;

    template <typename T>
    py::array_t<T> getAttributeNumpy(const string& name)
    {
        vector<T> res(size());
        for (unsigned int i = 0; i < size(); i++)
            res[i] = Field<T>::get(getItem(i), name);
        return py::array_t<T>(res.size(), res.data());
    }

    ObjId connectToSingle(const string& srcfield, const ObjId& tgt,
                          const string& tgtfield, const string& msgtype);

    ObjId connectToVec(const string& srcfield, const MooseVec& tgt,
                       const string& tgtfield, const string& msgtype);

    size_t id() const;

    // Iterator interface. Create copy of ObjId
    void generateIterator();
    const vector<ObjId>& objref() const;

private:
    ObjId oid_;
    std::string path_;
    vector<ObjId> objs_;
};

#endif /* end of include guard: MOOSE_VEC_H */
