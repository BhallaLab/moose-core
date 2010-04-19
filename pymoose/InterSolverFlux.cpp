#ifndef _pymoose_InterSolverFlux_cpp
#define _pymoose_InterSolverFlux_cpp
#include "InterSolverFlux.h"
using namespace pymoose;
const std::string InterSolverFlux::className_ = "InterSolverFlux";
InterSolverFlux::InterSolverFlux(Id id):PyMooseBase(id){}
InterSolverFlux::InterSolverFlux(std::string path):PyMooseBase(className_, path){}
InterSolverFlux::InterSolverFlux(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
InterSolverFlux::InterSolverFlux(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
InterSolverFlux::InterSolverFlux(const InterSolverFlux& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
InterSolverFlux::InterSolverFlux(const InterSolverFlux& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
InterSolverFlux::InterSolverFlux(const InterSolverFlux& src, std::string path):PyMooseBase(src, path){}
InterSolverFlux::InterSolverFlux(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
InterSolverFlux::InterSolverFlux(const Id& src, std::string path):PyMooseBase(src, path){}

InterSolverFlux::~InterSolverFlux(){}
const std::string& InterSolverFlux::getType(){ return className_; }
string InterSolverFlux::__get_method() const
{
    string method;
    get < string > (id_(), "method",method);
    return method;
}
void InterSolverFlux::__set_method( string method )
{
    set < string > (id_(), "method", method);
}
#endif
