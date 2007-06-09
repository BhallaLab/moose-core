#ifndef _pymoose_Kintegrator_cpp
#define _pymoose_Kintegrator_cpp
#include "Kintegrator.h"
const std::string Kintegrator::className = "Kintegrator";
Kintegrator::Kintegrator(Id id):PyMooseBase(id){}
Kintegrator::Kintegrator(std::string path):PyMooseBase(className, path){}
Kintegrator::Kintegrator(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Kintegrator::Kintegrator(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Kintegrator::~Kintegrator(){}
const std::string& Kintegrator::getType(){ return className; }
bool Kintegrator::__get_isInitiatilized() const
{
    bool isInitiatilized;
    get < bool > (Element::element(id_), "isInitiatilized",isInitiatilized);
    return isInitiatilized;
}
void Kintegrator::__set_isInitiatilized( bool isInitiatilized )
{
    set < bool > (Element::element(id_), "isInitiatilized", isInitiatilized);
}
// string Kintegrator::__get_method() const
// {
//     string method;
//     get < string > (Element::element(id_), "method",method);
//     return method;
// }
// void Kintegrator::__set_method( string method )
// {
//     set < string > (Element::element(id_), "method", method);
// }
string Kintegrator::imethod() const
{
    string method;
    get < string > (Element::element(id_), "method",method);
    return method;
}
string Kintegrator::imethod( string method )
{
    set < string > (Element::element(id_), "method", method);
    return method;    
}
#endif
