#ifndef _pymoose_Surface_cpp
#define _pymoose_Surface_cpp
#include "Surface.h"
using namespace pymoose;
const std::string Surface::className_ = "Surface";
Surface::Surface(Id id):Neutral(id){}
Surface::Surface(std::string path):Neutral(className_, path){}
Surface::Surface(std::string name, Id parentId):Neutral(className_, name, parentId){}
Surface::Surface(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Surface::Surface(const Surface& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Surface::Surface(const Surface& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Surface::Surface(const Surface& src, std::string path):Neutral(src, path){}
Surface::Surface(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Surface::Surface(const Id& src, std::string path):Neutral(src, path){}
Surface::~Surface(){}
const std::string& Surface::getType(){ return className_; }
double Surface::__get_volume() const
{
    double volume;
    get < double > (id_(), "volume",volume);
    return volume;
}
#endif
