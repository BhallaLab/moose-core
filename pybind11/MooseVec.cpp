/***
 *    Description:  vec api.
 *
 *        Created:  2020-04-01

 *         Author:  Dilawar Singh <dilawar.s.rajput@gmail.com>
 *        License:  MIT License
 */

#include <iomanip>

#include "../basecode/header.h"

using namespace std;

#include "../external/pybind11/include/pybind11/pybind11.h"
#include "../external/pybind11/include/pybind11/numpy.h"
#include "../external/pybind11/include/pybind11/stl.h"

namespace py = pybind11;

#include "../utility/strutil.h"

#include "Finfo.h"
#include "helper.h"
#include "pymoose.h"
#include "MooseVec.h"

MooseVec::MooseVec(const string& path, unsigned int n = 0,
                   const string& dtype = "Neutral")
    : path_(path)
{
    // If path is given and it does not exists, then create one. The old api
    // support it.
    oid_ = ObjId(path);
    if(oid_.bad())
        oid_ = mooseCreateFromPath(dtype, path, n);
}

MooseVec::MooseVec(const ObjId& oid) : oid_(oid), path_(oid.path())
{
}

MooseVec::MooseVec(const Id& id) : oid_(ObjId(id)), path_(id.path())
{
}

const string MooseVec::dtype() const
{
    return oid_.element()->cinfo()->name();
}

const size_t MooseVec::size() const
{
    if(oid_.element()->hasFields())
        return Field<unsigned int>::get(oid_, "numField");
    return oid_.element()->numData();
}

const string MooseVec::name() const
{
    return oid_.element()->getName();
}

const string MooseVec::path() const
{
    return path_;
}

ObjId MooseVec::parent() const
{
    return Neutral::parent(oid_);
}

vector<MooseVec> MooseVec::children() const
{
    vector<Id> children;
    Neutral::children(oid_.eref(), children);
    vector<MooseVec> res;
    std::transform(children.begin(), children.end(), res.begin(),
                   [](const Id& id) { return MooseVec(id); });
    return res;
}

unsigned int MooseVec::len()
{
    return (unsigned int)size();
}

ObjId MooseVec::getItem(const int index) const
{
    // Negative indexing.
    size_t i = (index < 0) ? size() + index : index;
    if(oid_.element()->hasFields())
        return getFieldItem(i);
    return getDataItem(i);
}

vector<ObjId> MooseVec::getItemRange(const py::slice& slice) const
{
    vector<ObjId> res;
    int start = 0, step = 1, stop = size();

    py::object pstart = slice.attr("start");
    if(!pstart.is(py::none()))
        start = pstart.cast<int>();

    py::object pstop = slice.attr("stop");
    if(!pstop.is(py::none()))
        stop = pstop.cast<int>();

    py::object pstep = slice.attr("step");
    if(!pstep.is(py::none()))
        step = pstep.cast<int>();

    for(int i = start; i < stop; i += step)
        res.push_back(getItem(i));
    return res;
}

ObjId MooseVec::getDataItem(const size_t i) const
{
    return ObjId(oid_.path(), i, oid_.fieldIndex);
}

ObjId MooseVec::getFieldItem(const size_t i) const
{
    return ObjId(oid_.path(), oid_.dataIndex, i);
}

py::object MooseVec::getAttribute(const string& name)
{
    // If type if double, int, bool etc, then return the numpy array. else
    // return the list of python object.
    auto cinfo = oid_.element()->cinfo();
    auto finfo = cinfo->findFinfo(name);
    if(!finfo) {
        auto fmap = __Finfo__::finfoNames(cinfo, "*");
        cerr << __func__ << ":: AttributeError: " << name
             << " is not found on path '" << oid_.path() << "'." << endl;
        cerr << finfoNotFoundMsg(cinfo) << endl;
        throw py::key_error(name + " is not found.");
    }

    auto rttType = finfo->rttiType();

    if(rttType == "double")
        return getAttributeNumpy<double>(name);
    if(rttType == "unsigned int")
        return getAttributeNumpy<unsigned int>(name);
    if(rttType == "int")
        return getAttributeNumpy<unsigned int>(name);

    vector<py::object> res(size());
    for(unsigned int i = 0; i < size(); i++)
        res[i] = getFieldGeneric(getItem((int)i), name);
    return py::cast(res);
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  API function. Set attribute on vector. This is the top-level
 * generic function.
 *
 * @Param name
 * @Param val
 *
 * @Returns
 */
/* ----------------------------------------------------------------------------*/
bool MooseVec::setAttribute(const string& name, const py::object& val)
{
    auto cinfo = oid_.element()->cinfo();
    auto finfo = cinfo->findFinfo(name);
    if(!finfo) {
        cerr << __func__ << ":: AttributeError: " << name
             << " is not found on path '" << oid_.path() << "'." << endl;
        cerr << finfoNotFoundMsg(cinfo) << endl;
        throw py::key_error(name + " is not found.");
    }

    auto rttType = finfo->rttiType();

    bool isVector = false;
    if(py::isinstance<py::iterable>(val) and(not py::isinstance<py::str>(val)))
        isVector = true;

    if(isVector) {
        if(rttType == "double")
            return setAttrOneToOne<double>(name, val.cast<vector<double>>());
        if(rttType == "unsigned int")
            return setAttrOneToOne<unsigned int>(
                name, val.cast<vector<unsigned int>>());
    } else {
        if(rttType == "double")
            return setAttrOneToAll<double>(name, val.cast<double>());
        if(rttType == "unsigned int")
            return setAttrOneToAll<unsigned int>(name,
                                                 val.cast<unsigned int>());
    }

    py::print("Not implemented yet.", name, "val:", val);
    throw runtime_error(__func__ + string("::NotImplementedError."));
}

ObjId MooseVec::connectToSingle(const string& srcfield, const ObjId& tgt,
                                const string& tgtfield, const string& msgtype)
{
    return shellConnect(oid_, srcfield, tgt, tgtfield, msgtype);
}

ObjId MooseVec::connectToVec(const string& srcfield, const MooseVec& tgt,
                             const string& tgtfield, const string& msgtype)
{
    if(size() != tgt.size())
        throw runtime_error(
            "Length mismatch. Source vector size is " + to_string(size()) +
            " but the target vector size is " + to_string(tgt.size()));
    return shellConnect(oid_, srcfield, tgt.obj(), tgtfield, msgtype);
}

const ObjId& MooseVec::obj() const
{
    return oid_;
}

vector<ObjId> MooseVec::objs() const
{
    vector<ObjId> items;
    for(size_t i = 0; i < size(); i++)
        items.push_back(ObjId(oid_.path(), i, 0));
    return items;
}

size_t MooseVec::id() const
{
    return oid_.id.value();
}

void MooseVec::generateIterator()
{
    objs_.resize(size());
    for(size_t i = 0; i < size(); i++)
        objs_[i] = getItem((int)i);
}

const vector<ObjId>& MooseVec::objref() const
{
    return objs_;
}
