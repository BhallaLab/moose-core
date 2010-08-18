#ifndef _pymoose_GssaStoich_cpp
#define _pymoose_GssaStoich_cpp
#include "GssaStoich.h"
using namespace pymoose;
const std::string GssaStoich::className_ = "GssaStoich";
GssaStoich::GssaStoich(Id id):Stoich(id){}
GssaStoich::GssaStoich(std::string path):Stoich(className_, path){}
GssaStoich::GssaStoich(std::string name, Id parentId):Stoich(className_, name, parentId){}
GssaStoich::GssaStoich(std::string name, PyMooseBase& parent):Stoich(className_, name, parent){}
GssaStoich::GssaStoich(const GssaStoich& src, std::string objectName, PyMooseBase& parent):Stoich(src, objectName, parent){}
GssaStoich::GssaStoich(const GssaStoich& src, std::string objectName, Id& parent):Stoich(src, objectName, parent){}
GssaStoich::GssaStoich(const GssaStoich& src, std::string path):Stoich(src, path){}
GssaStoich::GssaStoich(const Id& src, std::string name, Id& parent):Stoich(src, name, parent){}
GssaStoich::GssaStoich(const Id& src, std::string path):Stoich(src, path){}
GssaStoich::~GssaStoich(){}
const std::string& GssaStoich::getType(){ return className_; }
string GssaStoich::__get_method() const
{
    string method;
    get < string > (id_(), "method",method);
    return method;
}
void GssaStoich::__set_method( string method )
{
    set < string > (id_(), "method", method);
}
string GssaStoich::__get_path() const
{
    string path;
    get < string > (id_(), "path",path);
    return path;
}
void GssaStoich::__set_path( string path )
{
    set < string > (id_(), "path", path);
}
#endif
