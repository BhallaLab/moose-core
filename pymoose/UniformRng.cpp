#ifndef _pymoose_UniformRng_cpp
#define _pymoose_UniformRng_cpp
#include "UniformRng.h"
using namespace pymoose;

const std::string UniformRng::className = "UniformRng";
UniformRng::UniformRng(Id id):RandGenerator(id){}
UniformRng::UniformRng(std::string path):RandGenerator(className, path){}
UniformRng::UniformRng(std::string name, Id parentId):RandGenerator(className, name, parentId){}
UniformRng::UniformRng(std::string name, PyMooseBase& parent):RandGenerator(className, name, parent){}
UniformRng::UniformRng(const UniformRng& src, std::string objectName,  PyMooseBase& parent):RandGenerator(src, objectName, parent){}

UniformRng::UniformRng(const UniformRng& src, std::string objectName, Id& parent):RandGenerator(src, objectName, parent){}
UniformRng::UniformRng(const UniformRng& src, std::string path):RandGenerator(src, path)
{
}

UniformRng::UniformRng(const Id& src, string name, Id& parent):RandGenerator(src, name, parent)
{
}

UniformRng::~UniformRng(){}
const std::string& UniformRng::getType(){ return className; }
double UniformRng::__get_mean() const
{
    double mean;
    get < double > (id_(), "mean",mean);
    return mean;
}
double UniformRng::__get_variance() const
{
    double variance;
    get < double > (id_(), "variance",variance);
    return variance;
}
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
