#ifndef _pymoose_UniformRng_cpp
#define _pymoose_UniformRng_cpp
#include "UniformRng.h"
using namespace pymoose;

const std::string UniformRng::className_ = "UniformRng";
UniformRng::UniformRng(Id id):RandGenerator(id){}
UniformRng::UniformRng(std::string path):RandGenerator(className_, path){}
UniformRng::UniformRng(std::string name, Id parentId):RandGenerator(className_, name, parentId){}
UniformRng::UniformRng(std::string name, PyMooseBase& parent):RandGenerator(className_, name, parent){}
UniformRng::UniformRng(const UniformRng& src, std::string objectName,  PyMooseBase& parent):RandGenerator(src, objectName, parent){}

UniformRng::UniformRng(const UniformRng& src, std::string objectName, Id& parent):RandGenerator(src, objectName, parent){}
UniformRng::UniformRng(const UniformRng& src, std::string path):RandGenerator(src, path)
{
}
UniformRng::UniformRng(const Id& src, std::string path):RandGenerator(src, path)
{
}

UniformRng::UniformRng(const Id& src, string name, Id& parent):RandGenerator(src, name, parent)
{
}

UniformRng::~UniformRng(){}
const std::string& UniformRng::getType(){ return className_; }
double UniformRng::__get_min() const
{
    double min;
    get < double > (id_(), "min",min);
    return min;
}
void UniformRng::__set_min( double min )
{
    set < double > (id_(), "min", min);
}
double UniformRng::__get_max() const
{
    double max;
    get < double > (id_(), "max",max);
    return max;
}
void UniformRng::__set_max( double max )
{
    set < double > (id_(), "max", max);
}
#endif
