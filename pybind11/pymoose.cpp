// =====================================================================================
//
//       Filename:  pymoose.cpp
//
//    Description:  Python bindings generated using PyBind11.
//
//         Author:  Dilawar Singh (), dilawar.s.rajput@gmail.com
//   Organization:  NCBS Bangalore
//
// =====================================================================================

#include <map>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>
#include <functional>

#include "../external/pybind11/include/pybind11/pybind11.h"
#include "../external/pybind11/include/pybind11/stl.h"
#include "../external/pybind11/include/pybind11/numpy.h"

namespace py = pybind11;
using namespace std;
using namespace pybind11::literals;

#include "../basecode/global.h"
#include "../basecode/header.h"
#include "../builtins/Variable.h"
#include "../randnum/randnum.h"
#include "../shell/Neutral.h"
#include "../shell/Shell.h"
#include "../shell/Wildcard.h"
#include "../utility/strutil.h"
#include "Finfo.h"
#include "MooseVec.h"
#include "helper.h"
#include "pymoose.h"

Id initModule(py::module &m)
{
    return initShell();
}

bool setFieldGeneric(const ObjId &oid, const string &fieldName,
                     const py::object &val)
{
    auto cinfo = oid.element()->cinfo();
    auto finfo = cinfo->findFinfo(fieldName);
    if(!finfo) {
        throw runtime_error(fieldName + " is not found on " + oid.path());
        return false;
    }

    auto fieldType = finfo->rttiType();

    // Remove any space in fieldType
    fieldType.erase(
        std::remove_if(fieldType.begin(), fieldType.end(), ::isspace),
        fieldType.end());

    if(fieldType == "double")
        return Field<double>::set(oid, fieldName, val.cast<double>());
    if(fieldType == "vector<double>")
        return Field<vector<double>>::set(oid, fieldName,
                                          val.cast<vector<double>>());
    if(fieldType == "float")
        return Field<float>::set(oid, fieldName, val.cast<float>());
    if(fieldType == "unsignedint")
        return Field<unsigned int>::set(oid, fieldName,
                                        val.cast<unsigned int>());
    if(fieldType == "unsignedlong")
        return Field<unsigned long>::set(oid, fieldName,
                                         val.cast<unsigned long>());
    if(fieldType == "int")
        return Field<int>::set(oid, fieldName, val.cast<int>());
    if(fieldType == "bool")
        return Field<bool>::set(oid, fieldName, val.cast<bool>());
    if(fieldType == "string")
        return Field<string>::set(oid, fieldName, val.cast<string>());
    if(fieldType == "vector<string>")
        return Field<vector<string>>::set(oid, fieldName,
                                          val.cast<vector<string>>());
    if(fieldType == "char")
        return Field<char>::set(oid, fieldName, val.cast<char>());
    if(fieldType == "vector<ObjId>")
        return Field<vector<ObjId>>::set(oid, fieldName,
                                         val.cast<vector<ObjId>>());
    if(fieldType == "ObjId")
        return Field<ObjId>::set(oid, fieldName, val.cast<ObjId>());
    // if (fieldType == "Id") {
    //    // NB: Note that we cast to ObjId here and not to Id.
    //    return Field<Id>::set(oid.id, fieldName, val.cast<ObjId>());
    //}
    if(fieldType == "Id") {
        // NB: Handle MooseVec as well. Note that we send ObjId to the set
        // function. The C++ implicit conversion takes care of the rest.
        // Id tgt;
        if(py::isinstance<MooseVec>(val)) {
            auto tgt = Id(val.cast<MooseVec>().obj());
            return Field<Id>::set(oid.id, fieldName, tgt);
        } else {
            auto tgt = Id(val.cast<ObjId>());
            return Field<Id>::set(oid.id, fieldName, tgt);
        }
    }
    if(fieldType == "vector<double>") {
        // NB: Note that we cast to ObjId here and not to Id.
        return Field<vector<double>>::set(oid.id, fieldName,
                                          val.cast<vector<double>>());
    }
    if(fieldType == "vector<vector<double>>") {
        // NB: Note that we cast to ObjId here and not to Id.
        return Field<vector<vector<double>>>::set(
            oid.id, fieldName, val.cast<vector<vector<double>>>());
    }
    if(fieldType == "Variable")
        if(fieldType == "Variable")
            return Field<Variable>::set(oid, fieldName, val.cast<Variable>());

    throw runtime_error("NotImplemented::setField: '" + fieldName +
                        "' with value type '" + fieldType + "'.");
    return false;
}

py::object getFieldGeneric(const ObjId &oid, const string &fieldName)
{
    auto cinfo = oid.element()->cinfo();
    auto finfo = cinfo->findFinfo(fieldName);

    if(!finfo) {
        throw py::key_error(fieldName + " is not found on '" + oid.path() +
                            "'.");
    }

    string finfoType = cinfo->getFinfoType(finfo);

    // Things are very compilcated here. There return object from this function
    // can be of different types: a simple value (ValueFinfo), list, dict or
    // DestFinfo setter which is a function.
    if(finfoType == "ValueFinfo")
        return __Finfo__::getFieldValue(oid, finfo);
    else if(finfoType == "FieldElementFinfo") {
        // This is a Finfo
        return py::cast(__Finfo__(oid, finfo, "FieldElementFinfo"));
    } else if(finfoType == "LookupValueFinfo") {
        // This is a function.
        return py::cast(__Finfo__(oid, finfo, "LookupValueFinfo"));
    } else if(finfoType == "DestFinfo") {
        // Return a setter function.
        // It can be used to set field on DestFinfo.
        return __Finfo__::getDestFinfoSetterFunc(oid, finfo);
    }

    throw runtime_error("getFieldGeneric::NotImplemented : " + fieldName +
                        " with rttType " + finfo->rttiType() + " and type: '" +
                        finfoType + "'");
    return pybind11::none();
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  MOOSE extension module _moose.so.
 */
/* ----------------------------------------------------------------------------*/
PYBIND11_MODULE(_moose, m)
{
    // py::options options;
    // options.disable_function_signatures();

    m.doc() = R"moosedoc(
    pyMOOSE: Multiscale Object-Oriented Simulation Environment.
    )moosedoc";

    initModule(m);

    // A thin wrapper around Id from ../basecode/Id.h .
    py::class_<Id>(m, "__id__")
        .def(py::init<>())
        .def(py::init<unsigned int>())
        .def(py::init<const string &>())
        .def(py::init<const ObjId &>())
        // properties
        .def_property_readonly("numIds", &Id::numIds)
        .def_property_readonly("path", &Id::path)
        .def_property_readonly(
             "name", [](const Id &id) { return id.element()->getName(); })
        .def_property_readonly("id", &Id::value)
        .def("__getitem__", [](const Id &id, size_t i) { return ObjId(id, i); })
        .def_property_readonly("cinfo",
                               [](Id &id) { return id.element()->cinfo(); },
                               py::return_value_policy::reference)
        .def_property_readonly(
             "type", [](Id &id) { return id.element()->cinfo()->name(); })
        .def("__repr__", [](const Id &id) {
             return "<Id id=" + std::to_string(id.value()) + " path=" +
                    id.path() + " class=" + id.element()->cinfo()->name() + ">";
         })
        /**
         *  Override __eq__ etc.
         */
        .def("__eq__", [](const Id &a, const Id &b) { return a == b; })
        .def("__ne__", [](const Id &a, const Id &b) { return a != b; })
        .def("__hash__", &Id::value)

        // Id attributes are same as ObjItem attributes.
        .def("__getattr__", [](const Id &id, const string &key) {
             return getFieldGeneric(ObjId(id), key);
         })
        .def("__setattr__",
             [](const Id &id, const string &key, const py::object &val) {
             return setFieldGeneric(ObjId(id), key, val);
         });

    // This is a wrapper around Field::get  and LookupField::get which may
    // return simple values or vector. Python scripts expect LookupField to
    // return either list of dict which can be queried by key and index. This
    // class bind both __getitem__ to the getter function call.
    // Note that both a.isA["Compartment"] and a.isA("Compartment") are valid
    // now.
    py::class_<__Finfo__>(m, "mfield", py::dynamic_attr())
        .def(py::init<const ObjId &, const Finfo *, const char *>())
        // Only for FieldElementFinfos
        .def("__len__", &__Finfo__::getNumField)
        .def("__call__", &__Finfo__::operator())
        .def("__getitem__", &__Finfo__::getItem)
        .def("__setitem__", &__Finfo__::setItem)
        .def("__setattr__",
             [](__Finfo__ &f, const string &key, const py::object &val) {
             // FIXME: `num` is a special case here.
             if(key == "num")
                 return f.setNumField(val.cast<unsigned int>());
             return f.getMooseVecPtr()->setAttribute(key, val);
         })
        .def("__getattr__", [](__Finfo__ &f, const string &key) {
            // FIXME: `num` is a special case.
             if(key == "num")
                 return py::cast(f.getNumField());
             return f.getMooseVecPtr()->getAttribute(key);
         })
        .def_property_readonly("type", &__Finfo__::type)
        .def_property_readonly(
             "vec", [](__Finfo__ &finfo) { return finfo.getMooseVecPtr(); },
             py::return_value_policy::reference_internal);

    /**
     * @name ObjId. It is a base of all other moose objects.
     * @{ */
    /**  @} */
    py::class_<ObjId>(m, "melement", py::dynamic_attr())
        .def(py::init<>())
        .def(py::init<Id>())
        .def(py::init<Id, unsigned int>())
        .def(py::init<Id, unsigned int, unsigned int>())
        .def(py::init<const string &>())
        // Custom constructor.
        .def(py::init([](const ObjId &oid) {
             return ObjId(oid.id, oid.dataIndex, oid.fieldIndex);
         }))
        //---------------------------------------------------------------------
        //  Readonly properties.
        //---------------------------------------------------------------------
        .def_property_readonly("vec",
                               [](const ObjId &oid) { return MooseVec(oid); })
        .def_property_readonly(
             "parent", [](const ObjId &oid) { return Neutral::parent(oid); })
        .def_property_readonly(
             "name", [](const ObjId &oid) { return oid.element()->getName(); })
        .def_property_readonly("className", [](const ObjId &oid) {
             return oid.element()->cinfo()->name();
         })
        .def_property_readonly("id", [](ObjId & oid)->Id { return oid.id; })
        .def_property_readonly("dataIndex",
                               [](ObjId &oid) { return oid.dataIndex; })
        .def_property_readonly("fieldIndex",
                               [](ObjId &oid) { return oid.fieldIndex; })
        .def_property_readonly(
             "type", [](ObjId &oid) { return oid.element()->cinfo()->name(); })

        /**
         *  Override __eq__ etc.
         */
        .def("__eq__", [](const ObjId &a, const ObjId &b) { return a == b; })
        .def("__ne__", [](const ObjId &a, const ObjId &b) { return a != b; })
        .def("__hash__", [](const ObjId &oid) { return oid.id.value(); })

        /**
         * Attributes.
         */
        .def("getField", &getFieldGeneric)
        .def("setField", &setFieldGeneric)
        .def("__getattr__", &getFieldGeneric)
        .def("__setattr__", &setFieldGeneric)

        //---------------------------------------------------------------------
        //  Connect
        //---------------------------------------------------------------------
        .def("connect", &shellConnect, "srcfield"_a, "dest"_a, "destfield"_a,
             "msgtype"_a = "Single")
        .def("connect", &shellConnectToVec, "srcfield"_a, "dest"_a,
             "destfield"_a, "msgtype"_a = "Single")
        //---------------------------------------------------------------------
        //  Extra
        //---------------------------------------------------------------------
        .def("__repr__", [](const ObjId &oid) {
             return "<moose." + oid.element()->cinfo()->name() + " id=" +
                    std::to_string(oid.id.value()) + " dataIndex=" +
                    to_string(oid.eref().dataIndex()) + " path=" + oid.path() +
                    ">";
         })
        ;

    //// Cinfo.
    //py::class_<Cinfo>(m, "__Cinfo__")
    //    .def(py::init<>())
    //    .def_property_readonly("name", &Cinfo::name)
    //    .def_property_readonly("finfoMap", &Cinfo::finfoMap,
    //                           py::return_value_policy::reference_internal)
    //    // .def("findFinfo", &Cinfo::findFinfoWrapper)
    //    .def("baseCinfo", &Cinfo::baseCinfo,
    //         py::return_value_policy::reference_internal)
    //    .def("isA", &Cinfo::isA, py::return_value_policy::reference_internal);

    // Vec class.
    py::class_<MooseVec>(m, "vec")
        .def(py::init<const string &, unsigned int, const string &>(), "path"_a,
             "n"_a = 1, "dtype"_a = "Neutral")  // Default
        .def(py::init<const ObjId &>())
        .def("__eq__", [](const MooseVec &a,
                          const MooseVec &b) { return a.obj() == b.obj(); })
        .def("__ne__", [](const MooseVec &a,
                          const MooseVec &b) { return a.obj() != b.obj(); })
        .def("__len__", &MooseVec::len)
        .def("__iter__",
             [](MooseVec &v) {
                 // Generate an iterator which is a vector<ObjId>. And then
                 // pass the reference to the objects.
                 v.generateIterator();
                 return py::make_iterator(v.objref().begin(), v.objref().end());
             },
             py::keep_alive<0, 1>())
        .def("__getitem__", &MooseVec::getItem)
        .def("__getitem__", &MooseVec::getItemRange)

        // Templated function won't work here. The first one is always called.
        .def("__getattr__", &MooseVec::getAttribute)
        .def("__setattr__", &MooseVec::setAttribute)
        .def("__repr__", [](const MooseVec & v)->string {
             return "<moose.vec class=" + v.dtype() + " path=" + v.path() +
                    " id=" + std::to_string(v.id()) + " size=" +
                    std::to_string(v.size()) + ">";
         })
        // This is to provide old API support. Some scripts use .vec even on a
        // vec to get a vec. So silly or so Zen?!
        .def_property_readonly("vec", [](const MooseVec &vec) { return &vec; },
                               py::return_value_policy::reference_internal)
        .def_property_readonly("type",
                               [](const MooseVec &v) { return "moose.vec"; })
        .def("connect", &MooseVec::connectToSingle)
        .def("connect", &MooseVec::connectToVec)

        // Thi properties are not vectorised.
        .def_property_readonly("parent", &MooseVec::parent)
        .def_property_readonly("children", &MooseVec::children)
        .def_property_readonly("name", &MooseVec::name)
        .def_property_readonly("path", &MooseVec::path)
        // Wrapped object.
        .def_property_readonly("objid", &MooseVec::obj);

    // Module functions.
    m.def("getShell",
          []() { return reinterpret_cast<Shell *>(Id().eref().data()); },
          py::return_value_policy::reference);

    m.def("seed", [](py::object& a) { moose::mtseed(a.cast<int>()); });
    m.def("rand", [](double a, double b) { return moose::mtrand(a, b); },
          "a"_a = 0, "b"_a = 1);
    // This is a wrapper to Shell::wildcardFind. The python interface must
    // override it.
    m.def("wildcardFind", &wildcardFind2);
    m.def("delete", &mooseDeleteStr);
    m.def("delete", &mooseDeleteObj);
    m.def("create", &mooseCreateFromPath);
    m.def("create", &mooseCreateFromObjId);
    m.def("create", &mooseCreateFromId);
    m.def("create", &mooseCreateFromMooseVec);
    m.def("move", &mooseMove<ObjId, ObjId>);
    m.def("move", &mooseMove<ObjId, string>);
    m.def("move", &mooseMove<string, ObjId>);
    m.def("move", &mooseMove<string, string>);
    m.def("reinit", &mooseReinit);
    m.def("start", &mooseStart, "runtime"_a, "notify"_a = false);
    m.def("stop", &mooseStop);
    m.def("element", &mooseObjIdPath);
    m.def("element", &mooseObjIdObj);
    m.def("element", &mooseObjIdId);
    m.def("element", &mooseObjIdField);
    m.def("element", &mooseObjIdMooseVec);
    m.def("exists", &mooseExists);
    m.def("getCwe", &mooseGetCwe);
    m.def("setCwe", &mooseSetCwe);
    m.def("setClock", &mooseSetClock);
    m.def("useClock", &mooseUseClock);
    m.def("loadModelInternal", &loadModelInternal);
    m.def("getFieldNames", &mooseGetFieldNames);
    m.def("getField",
          [](const ObjId &oid, const string &fieldName, const string &ftype) {
              // ftype is not needed anymore.
              return getFieldGeneric(oid, fieldName);
          },
          "el"_a, "fieldname"_a, "ftype"_a = "");
    m.def("getFieldDict", &mooseGetFieldDict, "className"_a,
          "finfoType"_a = "");
    m.def("copy", &mooseCopy, "orig"_a, "newParent"_a, "newName"_a, "num"_a = 1,
          "toGlobal"_a = false, "copyExtMsgs"_a = false);

    // This is better handled by moose.py. Later we can move it back here.
    // m.def("connect", &mooseConnect, "src"_a, "srcfield"_a, "dest"_a,
    //      "destfield"_a, "msgtype"_a = "Single");

    // Attributes.
    m.attr("NA") = NA;
    m.attr("PI") = PI;
    m.attr("FaradayConst") = FaradayConst;
    m.attr("GasConst") = GasConst;
    m.attr("__version__") = MOOSE_VERSION;
}
