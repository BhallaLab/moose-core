// =====================================================================================
//
//       Filename:  helper.h
//
//    Description:
//
//        Version:  1.0
//        Created:  03/22/2020 09:06:16 PM
//       Revision:  none
//       Compiler:  g++
//
//         Author:  Dilawar Singh (), dilawar.s.rajput@gmail.com
//   Organization:  NCBS Bangalore
//
// =====================================================================================

#ifndef HELPER_H
#define HELPER_H

#include "../shell/Shell.h"
#include "../utility/strutil.h"

#include "MooseVec.h"
#include "Finfo.h"

namespace py = pybind11;
using namespace std;

ObjId mooseGetCweId();
py::object mooseGetCwe();

void mooseSetCwe(const py::object& arg);
// void mooseSetCwe(const ObjId& oid);

inline Shell* getShellPtr(void)
{
    return reinterpret_cast<Shell*>(Id().eref().data());
}

Id initShell();

bool mooseExists(const string& path);

#if 0
void mooseMoveId(const Id& a, const ObjId& b);
void mooseMoveObjId(const ObjId& a, const ObjId& b);
#endif

template <typename P = ObjId, typename Q = ObjId>
inline void mooseMove(const P& src, const Q& tgt)
{
    getShellPtr()->doMove(Id(src), ObjId(tgt));
}

inline ObjId mooseObjIdPath(const string& p)
{
    // handle relative path.
    string path(p);

    // If path is a relative path.
    if(p[0] != '/') {
        string cwepath(mooseGetCweId().path());
        if(cwepath.back() != '/')
            cwepath.push_back('/');
        path = cwepath + p;
    }
    ObjId oid(path);
    if(oid.bad()) {
        cerr << "moose.element: " << path << " does not exist!" << endl;
    }
    return oid;
}

inline ObjId mooseObjIdObj(const ObjId& obj)
{
    return obj;
}

inline ObjId mooseObjIdId(const Id& id)
{
    return ObjId(id);
}

inline ObjId mooseObjIdField(const __Finfo__& finfo)
{
    return finfo.getObjId();
}

inline ObjId mooseObjIdMooseVec(const MooseVec& vec)
{
    return vec.obj();
}

inline ObjId mooseCreateFromPath(const string type, const string& p,
                                 unsigned int numdata)
{

    // NOTE: This function is bit costly because of regex use. One can replace
    // it with bit more efficient one if required.
    auto path = moose::normalizePath(p);

    if(path.at(0) != '/') {
        string cwe = mooseGetCweId().path();
        if(cwe.back() != '/')
            cwe += '/';
        path = cwe + path;
    }

    // Split into dirname and basename component.
    auto pp = moose::splitPath(path);
    string name(pp.second);
    if(name.back() == ']')
        name = name.substr(0, name.find_last_of('['));

    // Check if parent exists.
    auto parent = ObjId(pp.first);
    if(parent.bad()) {
        throw py::key_error("Parent '" + pp.first +
                            "' is not found. Not creating...");
        return Id();
    }

    // If path exists and user is asking for the same type then return the
    // underlying object else raise an exception.
    auto oid = ObjId(path);
    if(not oid.bad()) {
        if(oid.element()->cinfo()->name() == type)
            return oid;
        else
            throw runtime_error("An object with path'" + path +
                                "' already "
                                "exists. Use moose.element to access it.");
    }

    return getShellPtr()->doCreate2(type, ObjId(pp.first), name, numdata);
}

inline ObjId mooseCreateFromObjId(const string& type, const ObjId& oid,
                                  unsigned int numData)
{
    return oid;
}

inline ObjId mooseCreateFromId(const string& type, const Id& id,
                               unsigned int numData)
{
    return ObjId(id);
}

inline ObjId mooseCreateFromMooseVec(const string& type, const MooseVec& vec,
                                     unsigned int numData)
{
    return vec.obj();
}

ObjId loadModelInternal(const string& fname, const string& modelpath,
                        const string& solverclass);

ObjId getElementField(const ObjId objid, const string& fname);

ObjId getElementFieldItem(const ObjId& objid, const string& fname,
                          unsigned int index);

// Connect using doConnect
ObjId shellConnect(const ObjId& src, const string& srcField, const ObjId& tgt,
                   const string& tgtField, const string& msgType);

ObjId shellConnectToVec(const ObjId& src, const string& srcField,
                        const MooseVec& tgt, const string& tgtField,
                        const string& msgType);

inline bool mooseDeleteObj(const ObjId& oid)
{
    return getShellPtr()->doDelete(oid);
}

inline bool mooseDeleteStr(const string& path)
{
    auto o = ObjId(path);
    if(o.bad())
        throw pybind11::value_error("Path '" + path + "' does not exists.");
    return getShellPtr()->doDelete(ObjId(path));
}

MooseVec mooseCopy(const py::object& orig, const py::object& newParent,
                   string newName, unsigned int n, bool toGlobal,
                   bool copyExtMsgs);

ObjId mooseCreate(const string type, const string& path,
                  unsigned int numdata = 1);

void mooseSetClock(const unsigned int clockId, double dt);

void mooseUseClock(size_t tick, const string& path, const string& field);

// API.
map<string, string> mooseGetFieldDict(const string& className,
                                      const string& finfoType);

// Internal.
map<string, Finfo*> getFieldDict(const string& className,
                                 const string& finfoType);
map<string, Finfo*> innerGetFieldDict(const Cinfo* cinfo,
                                      const string& finfoType);

void mooseReinit();

void handleKeyboardInterrupts(int signum);

void mooseStart(double runtime, bool notify);

void mooseStop();

py::cpp_function getPropertyDestFinfo(const ObjId& oid, const Finfo* finfo);

vector<string> mooseGetFieldNames(const string& className,
                                  const string& finfoType);

string finfoNotFoundMsg(const Cinfo* cinfo);

bool mooseIsRunning();

string mooseClassDoc(const string& classname);

string mooseDoc(const string& string);

vector<string> mooseLe(const ObjId& obj);

string mooseShowMsg(const ObjId& obj);

vector<ObjId> mooseListMsg(const ObjId& obj);

#endif /* end of include guard: HELPER_H */
