#ifndef _pymoose_Surface_cpp
#define _pymoose_Surface_cpp
#include "Surface.h"
using namespace pymoose;
const std::string Surface::className_ = "Surface";
Surface::Surface(Id id):PyMooseBase(id){}
Surface::Surface(std::string path):PyMooseBase(className_, path){}
Surface::Surface(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
Surface::Surface(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
Surface::Surface(const Surface& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
Surface::Surface(const Surface& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
Surface::Surface(const Surface& src, std::string path):PyMooseBase(src, path){}
Surface::Surface(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
Surface::~Surface(){}
const std::string& Surface::getType(){ return className_; }
double Surface::__get_volume() const
{
    double volume;
    get < double > (id_(), "volume",volume);
    return volume;
}
#endif
