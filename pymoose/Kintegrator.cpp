#ifndef _pymoose_Kintegrator_cpp
#define _pymoose_Kintegrator_cpp
#include "Kintegrator.h"
using namespace pymoose;
const std::string Kintegrator::className_ = "Kintegrator";
Kintegrator::Kintegrator(Id id):Neutral(id){}
Kintegrator::Kintegrator(std::string path):Neutral(className_, path){}
Kintegrator::Kintegrator(std::string name, Id parentId):Neutral(className_, name, parentId){}
Kintegrator::Kintegrator(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Kintegrator::Kintegrator(const Kintegrator& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Kintegrator::Kintegrator(const Kintegrator& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Kintegrator::Kintegrator(const Kintegrator& src, std::string path):Neutral(src, path){}
Kintegrator::Kintegrator(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Kintegrator::Kintegrator(const Id& src, std::string path):Neutral(src, path){}
Kintegrator::~Kintegrator(){}
const std::string& Kintegrator::getType(){ return className_; }
bool Kintegrator::__get_isInitiatilized() const
{
    bool isInitiatilized;
    get < bool > (id_(), "isInitiatilized",isInitiatilized);
    return isInitiatilized;
}
string  Kintegrator::__get_method() const
{
    string method;
    get < string > (id_(), "method",method);
    return method;
}
void Kintegrator::__set_method( string method )
{
    set < string > (id_(), "method", method);
}
#endif
