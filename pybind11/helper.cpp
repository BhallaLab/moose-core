// =====================================================================================
//
//       Filename:  helper.cpp
//
//    Description: Helper functions.
//
//        Version:  1.0
//        Created:  03/22/2020 09:05:13 PM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Dilawar Singh (), dilawar.s.rajput@gmail.com
//   Organization:  NCBS Bangalore
//
// =====================================================================================

#include <memory>
#include <stdexcept>
#include <csignal>

#include "../external/pybind11/include/pybind11/functional.h"
#include "../external/pybind11/include/pybind11/numpy.h"
#include "../external/pybind11/include/pybind11/pybind11.h"
#include "../external/pybind11/include/pybind11/stl.h"

#include "../external/fmt-6.2.0/include/fmt/format-inl.h"

namespace py = pybind11;

// See
// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#binding-stl-containers
// #include "../external/pybind11/include/pybind11/stl_bind.h"

#include "../basecode/header.h"
#include "../basecode/global.h"

#include "../builtins/Variable.h"
#include "../mpi/PostMaster.h"
#include "../scheduling/Clock.h"
#include "../shell/Neutral.h"
#include "../shell/Shell.h"
#include "../shell/Wildcard.h"
#include "../utility/strutil.h"
#include "../randnum/randnum.h"

#include "helper.h"
#include "pymoose.h"
#include "Finfo.h"

using namespace std;

using namespace std;

Id initShell(void)
{
    Cinfo::rebuildOpIndex();

    Id shellId;

    Element* shelle =
        new GlobalDataElement(shellId, Shell::initCinfo(), "/", 1);

    Id clockId = Id::nextId();
    assert(clockId.value() == 1);
    Id classMasterId = Id::nextId();
    Id postMasterId = Id::nextId();

    Shell* s = reinterpret_cast<Shell*>(shellId.eref().data());
    s->setHardware(1, 1, 0);
    s->setShellElement(shelle);

    /// Sets up the Elements that represent each class of Msg.
    auto numMsg = Msg::initMsgManagers();

    new GlobalDataElement(clockId, Clock::initCinfo(), "clock", 1);
    new GlobalDataElement(classMasterId, Neutral::initCinfo(), "classes", 1);
    new GlobalDataElement(postMasterId, PostMaster::initCinfo(), "postmaster",
                          1);

    assert(shellId == Id());
    assert(clockId == Id(1));
    assert(classMasterId == Id(2));
    assert(postMasterId == Id(3));

    Shell::adopt(shellId, clockId, numMsg++);
    Shell::adopt(shellId, classMasterId, numMsg++);
    Shell::adopt(shellId, postMasterId, numMsg++);
    assert(numMsg == 10);  // Must be the same on all nodes.

    Cinfo::makeCinfoElements(classMasterId);
    return shellId;
}

/**
 * @brief Handle signal raised by user during simulation.
 *
 * @param signum
 */
void handleKeyboardInterrupts(int signum)
{
    LOG(moose::info, "Interrupt signal (" << signum << ") received.");
    // Get the shell and cleanup.
    Shell* shell = getShellPtr();
    shell->cleanSimulation();
    exit(signum);
}

/**
   Utility function to create objects from full path, dimensions
   and classname.
*/
ObjId createIdFromPath(string path, string type, unsigned int numData)
{
    path = moose::fix(path);

    Shell* pShell = reinterpret_cast<Shell*>(Id().eref().data());
    string parent_path;
    string name;

    string trimmed_path = moose::trim(path);

    auto pos = trimmed_path.rfind("/");
    if(pos != string::npos) {
        name = trimmed_path.substr(pos + 1);
        parent_path = trimmed_path.substr(0, pos);
    }
    else {
        name = trimmed_path;
    }
    // handle relative path
    if(trimmed_path[0] != '/') {
        string current_path = pShell->getCwe().path();
        if(current_path != "/")
            parent_path = current_path + "/" + parent_path;
        else
            parent_path = current_path + parent_path;
    }
    else if(parent_path.empty())
        parent_path = "/";

    ObjId parent_id(parent_path);
    if(parent_id.bad()) {
        string message = "Parent element does not exist: ";
        message += parent_path;
        throw std::runtime_error(message);
        return Id();
    }

    auto nId =
        pShell->doCreate(type, parent_id, string(name), numData, MooseGlobal);

    if(nId == Id() && trimmed_path != "/" && trimmed_path != "/root") {
        string message = "no such moose class : " + type;
        throw std::runtime_error(message);
    }

    return nId;
}

bool mooseExists(const string& path)
{
    return Id(path) != Id() || path == "/" || path == "/root";
}

ObjId loadModelInternal(const string& fname, const string& modelpath,
                        const string& solverclass = "")
{
    Id model;
    if(solverclass.empty()) {
        model = getShellPtr()->doLoadModel(fname, modelpath);
    }
    else {
        model = getShellPtr()->doLoadModel(fname, modelpath, solverclass);
    }

    if(model == Id()) {
        throw runtime_error("could not load model");
        return Id();
    }
    return ObjId(model);
}

ObjId getElementField(const ObjId objid, const string& fname)
{
    return ObjId(objid.path() + '/' + fname);
}

ObjId getElementFieldItem(const ObjId& objid, const string& fname,
                          unsigned int index)
{
    ObjId oid = getElementField(objid, fname);

    auto len = Field<unsigned int>::get(oid, "numField");
    assert(len >= 0);

    if(index >= len) {
        throw runtime_error(
            "ElementField.getItem: index out of bounds. "
            "Total elements=" +
            to_string(len) + ".");
        return ObjId();
    }

    // Negative indexing. Thanks Subha for hint.
    if(index < 0) {
        index += len;
    }
    if(index < 0) {
        throw runtime_error(
            "ElementField.getItem: invalid index: " + to_string(index) + ".");
        return ObjId();
    }
    return ObjId(oid.id, oid.dataIndex, index);
}

ObjId shellConnect(const ObjId& src, const string& srcField, const ObjId& tgt,
                   const string& tgtField, const string& msgType)
{
    return getShellPtr()->doAddMsg(msgType, src, srcField, tgt, tgtField);
}

ObjId shellConnectToVec(const ObjId& src, const string& srcField,
                        const MooseVec& tgt, const string& tgtField,
                        const string& msgType)
{
    return getShellPtr()->doAddMsg(msgType, src, srcField, tgt.obj(), tgtField);
}

#if 0
void mooseMoveId(const Id& a, const ObjId& b)
{
    getShellPtr()->doMove(a, b);
}

void mooseMoveObjId(const ObjId& a, const ObjId& b)
{
    getShellPtr()->doMove(a.id, b);
}

void mooseMoveObPath(const string& a, const ObjId& b)
{
    getShellPtr()->doMove(a.id, b);
}
#endif

void mooseSetClock(const unsigned int clockId, double dt)
{
    getShellPtr()->doSetClock(clockId, dt);
}

void mooseUseClock(size_t tick, const string& path, const string& field)
{
    getShellPtr()->doUseClock(path, field, tick);
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Current Working Element.
 *
 * @Returns  cwe.
 */
/* ----------------------------------------------------------------------------*/
ObjId mooseGetCweId()
{
    return getShellPtr()->getCwe();
}

py::object mooseGetCwe()
{
    return py::cast(mooseGetCweId());
}

void mooseSetCwe(const py::object& arg)
{
    if(py::isinstance<py::str>(arg))
        return getShellPtr()->setCwe(ObjId(arg.cast<string>()));
    return getShellPtr()->setCwe(arg.cast<ObjId>());
}

map<string, string> mooseGetFieldDict(const string& className,
                                      const string& finfoType = "*")
{
    map<string, string> res;
    for(const auto& v : getFieldDict(className, finfoType))
        res[v.first] = v.second->rttiType();
    return res;
}

map<string, Finfo*> getFieldDict(const string& className,
                                 const string& finfoType = "*")
{
    const Cinfo* cinfo = Cinfo::find(className);
    if(!cinfo) {
        cout << "Warning: Invalid class " << className << endl;
        return {};
    }
    return innerGetFieldDict(cinfo, finfoType);
}

map<string, Finfo*> innerGetFieldDict(const Cinfo* cinfo,
                                      const string& finfoType = "*")
{
    if(finfoType == "*")
        return cinfo->finfoMap();

    map<string, Finfo*> fieldDict;

    if(finfoType == "valueFinfo" || finfoType == "value") {
        for(unsigned int ii = 0; ii < cinfo->getNumValueFinfo(); ++ii) {
            auto* finfo = cinfo->getValueFinfo(ii);
            fieldDict[finfo->name()] = finfo;
        }
    }
    else if(finfoType == "srcFinfo" || finfoType == "src") {
        for(unsigned int ii = 0; ii < cinfo->getNumSrcFinfo(); ++ii) {
            auto* finfo = cinfo->getSrcFinfo(ii);
            fieldDict[finfo->name()] = finfo;
        }
    }
    else if(finfoType == "destFinfo" || finfoType == "dest") {
        for(unsigned int ii = 0; ii < cinfo->getNumDestFinfo(); ++ii) {
            auto* finfo = cinfo->getDestFinfo(ii);
            fieldDict[finfo->name()] = finfo;
        }
    }
    else if(finfoType == "lookupFinfo" || finfoType == "lookup") {
        for(unsigned int ii = 0; ii < cinfo->getNumLookupFinfo(); ++ii) {
            auto* finfo = cinfo->getLookupFinfo(ii);
            fieldDict[finfo->name()] = finfo;
        }
    }
    else if(finfoType == "sharedFinfo" || finfoType == "shared") {
        for(unsigned int ii = 0; ii < cinfo->getNumSrcFinfo(); ++ii) {
            auto* finfo = cinfo->getSrcFinfo(ii);
            fieldDict[finfo->name()] = finfo;
        }
    }
    else if(finfoType == "fieldElementFinfo" || finfoType == "field" ||
            finfoType == "fieldElement") {
        for(unsigned int ii = 0; ii < cinfo->getNumFieldElementFinfo(); ++ii) {
            auto* finfo = cinfo->getFieldElementFinfo(ii);
            fieldDict[finfo->name()] = finfo;
        }
    }
    return fieldDict;
}

void mooseReinit()
{
    getShellPtr()->doReinit();
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Register and signal handler and start the simulation. When ctrl+c
 * is pressed, stop, cleanup and exit.
 *
 * @Param runtime
 * @Param notify
 */
/* ----------------------------------------------------------------------------*/
void mooseStart(double runtime, bool notify = false)
{
    struct sigaction sigHandler;
    sigHandler.sa_handler = handleKeyboardInterrupts;
    sigemptyset(&sigHandler.sa_mask);
    sigHandler.sa_flags = 0;
    sigaction(SIGINT, &sigHandler, NULL);
    getShellPtr()->doStart(runtime, notify);
}

void mooseStop()
{
    getShellPtr()->doStop();
}

// Id is synonym with Id in previous binding.
MooseVec mooseCopy(const py::object& elem, const py::object& newParent,
                   string newName, unsigned int n = 1, bool toGlobal = false,
                   bool copyExtMsgs = false)
{
    Id orig = py::cast<Id>(elem);
    ObjId newp;
    if(py::isinstance<MooseVec>(newParent))
        newp = newParent.cast<MooseVec>().obj();
    else
        newp = newParent.cast<ObjId>();
    return MooseVec(
        getShellPtr()->doCopy(orig, newp, newName, n, toGlobal, copyExtMsgs));
}

/**
  Return a vector of field names of specified finfo type. This is from Subha.
*/
vector<string> mooseGetFieldNames(const string& className,
                                  const string& finfoType)
{
    vector<string> ret;
    const Cinfo* cinfo = Cinfo::find(className);
    if(!cinfo) {
        cerr << "Invalid class name." << endl;
        return ret;
    }

    if(finfoType == "valueFinfo" || finfoType == "value") {
        for(unsigned int ii = 0; ii < cinfo->getNumValueFinfo(); ++ii) {
            Finfo* finfo = cinfo->getValueFinfo(ii);
            ret.push_back(finfo->name());
        }
    }
    else if(finfoType == "srcFinfo" || finfoType == "src") {
        for(unsigned int ii = 0; ii < cinfo->getNumSrcFinfo(); ++ii) {
            Finfo* finfo = cinfo->getSrcFinfo(ii);
            ret.push_back(finfo->name());
        }
    }
    else if(finfoType == "destFinfo" || finfoType == "dest") {
        for(unsigned int ii = 0; ii < cinfo->getNumDestFinfo(); ++ii) {
            Finfo* finfo = cinfo->getDestFinfo(ii);
            ret.push_back(finfo->name());
        }
    }
    else if(finfoType == "lookupFinfo" || finfoType == "lookup") {
        for(unsigned int ii = 0; ii < cinfo->getNumLookupFinfo(); ++ii) {
            Finfo* finfo = cinfo->getLookupFinfo(ii);
            ret.push_back(finfo->name());
        }
    }
    else if(finfoType == "sharedFinfo" || finfoType == "shared") {
        for(unsigned int ii = 0; ii < cinfo->getNumSrcFinfo(); ++ii) {
            Finfo* finfo = cinfo->getSrcFinfo(ii);
            ret.push_back(finfo->name());
        }
    }
    else if(finfoType == "fieldElementFinfo" || finfoType == "fieldElement") {
        for(unsigned int ii = 0; ii < cinfo->getNumFieldElementFinfo(); ++ii) {
            Finfo* finfo = cinfo->getFieldElementFinfo(ii);
            ret.push_back(finfo->name());
        }
    }
    return ret;
}

string finfoNotFoundMsg(const Cinfo* cinfo)
{
    auto fmap = __Finfo__::finfoNames(cinfo, "*");
    stringstream ss;
    ss << "Available attributes:" << endl;
    for(size_t i = 0; i < fmap.size(); i++) {
        ss << setw(15) << fmap[i].first;
        if((i + 1) % 5 == 0)
            ss << endl;
    }
    return ss.str();
}

bool mooseIsRunning()
{
    return getShellPtr()->isRunning();
}

string fieldDocFormatted(const string& name, const Cinfo* cinfo,
                         const Finfo* finfo)
{
    return fmt::format("{0:<15}\n Type: {1:}\n{2}\n\n", name, finfo->rttiType(),
                       moose::textwrap(finfo->docs(), "  "));
}

string mooseClassFieldDoc(const Cinfo* cinfo, const string& ftype)
{
    stringstream ss;

    auto fmap = innerGetFieldDict(cinfo, ftype);

    for(const auto& v : fmap)
        ss << fieldDocFormatted(v.first, cinfo, v.second);

    // There are from base classes.
    const Cinfo* baseClassCinfo = cinfo->baseCinfo();
    while(baseClassCinfo) {
        auto baseFmap = innerGetFieldDict(baseClassCinfo, ftype);
        for(const auto& vv : baseFmap) {
            if(fmap.find(vv.first) == fmap.end()) {
                fmap[vv.first] = vv.second;
                ss << fieldDocFormatted(vv.first, baseClassCinfo, vv.second);
            }
        }
        baseClassCinfo = baseClassCinfo->baseCinfo();
    }
    return ss.str();
}

string mooseClassDoc(const string& className)
{
    stringstream ss;

    auto cinfo = Cinfo::find(className);
    if(!cinfo) {
        ss << "This class is not valid." << endl;
        return ss.str();
    }

    // ss << fmt::format(
    //     "{0:—^{2}}\n{1: ^{2}}{0:—^{2}}\n", "",
    //     moose::textwrap(
    //         "Following document is generated from the source code for quick "
    //         " reference. Go to [https://moose.readthedocs.io/en/latest/] for"
    //         " tutorials and examples."),
    //     70);
    // ss << cinfo->getDocs();

    // Documentation of Finfo.
    // auto finfos = cinfo->finfoMap();
    // for(const auto& item : finfos) {
    // ss << item.first << endl;
    // ss << moose::textwrap(item.second->docs(), "  ", 70) << endl;
    // }

    // ss << moose::boxed("Value Field");
    ss << mooseClassFieldDoc(cinfo, "value");

    // Collect all field from this class and the base classes.
    // for(unsigned int ii = 0; ii < cinfo->getNumValueFinfo(); ++ii) {
    //    auto* finfo = cinfo->getValueFinfo(ii);
    //    ss << fmt::format("{0}\n{1}\n\n", finfo->name(),
    //                      moose::textwrap(finfo->docs(), "  "));
    //}

    return ss.str();
}
