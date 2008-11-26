#ifndef _pymoose_Kintegrator_cpp
#define _pymoose_Kintegrator_cpp
#include "Kintegrator.h"
using namespace pymoose;
const std::string Kintegrator::className_ = "Kintegrator";
Kintegrator::Kintegrator(Id id):PyMooseBase(id){}
Kintegrator::Kintegrator(std::string path):PyMooseBase(className_, path){}
Kintegrator::Kintegrator(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
Kintegrator::Kintegrator(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
Kintegrator::Kintegrator(const Kintegrator& src, std::string objectName,  PyMooseBase& parent):PyMooseBase(src, objectName, parent){}

Kintegrator::Kintegrator(const Kintegrator& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
Kintegrator::Kintegrator(const Kintegrator& src, std::string path):PyMooseBase(src, path)
{
}

Kintegrator::Kintegrator(const Id& src, string name, Id& parent):PyMooseBase(src, name, parent)
{
}
Kintegrator::~Kintegrator(){}
const std::string& Kintegrator::getType(){ return className_; }
bool Kintegrator::__get_isInitiatilized() const
{
    bool isInitiatilized;
    get < bool > (id_(), "isInitiatilized",isInitiatilized);
    return isInitiatilized;
}
void Kintegrator::__set_isInitiatilized( bool isInitiatilized )
{
    set < bool > (id_(), "isInitiatilized", isInitiatilized);
}
// string Kintegrator::__get_method() const
// {
//     string method;
//     get < string > (id_(), "method",method);
//     return method;
// }
// void Kintegrator::__set_method( string method )
// {
//     set < string > (id_(), "method", method);
// }
string Kintegrator::imethod() const
{
    string method;
    get < string > (id_(), "method",method);
    return method;
}
string Kintegrator::imethod( string method )
{
    set < string > (id_(), "method", method);
    return method;    
}
#endif
