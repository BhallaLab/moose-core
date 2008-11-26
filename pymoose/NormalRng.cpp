#ifndef _pymoose_NormalRng_cpp
#define _pymoose_NormalRng_cpp
#include "NormalRng.h"
using namespace pymoose;
const std::string NormalRng::className_ = "NormalRng";
NormalRng::NormalRng(Id id):RandGenerator(id){}
NormalRng::NormalRng(std::string path):RandGenerator(className_, path){}
NormalRng::NormalRng(std::string name, Id parentId):RandGenerator(className_, name, parentId){}
NormalRng::NormalRng(std::string name, PyMooseBase& parent):RandGenerator(className_, name, parent){}
NormalRng::NormalRng(const NormalRng& src, std::string objectName,  PyMooseBase& parent):RandGenerator(src, objectName, parent){}

NormalRng::NormalRng(const NormalRng& src, std::string objectName, Id& parent):RandGenerator(src, objectName, parent){}
NormalRng::NormalRng(const NormalRng& src, std::string path):RandGenerator(src, path)
{
}

NormalRng::NormalRng(const Id& src, string name, Id& parent):RandGenerator(src, name, parent)
{
}
NormalRng::~NormalRng(){}
const std::string& NormalRng::getType(){ return className_; }
double NormalRng::__get_mean() const
{
    double mean;
    get < double > (id_(), "mean",mean);
    return mean;
}
void NormalRng::__set_mean( double mean )
{
    set < double > (id_(), "mean", mean);
}
double NormalRng::__get_variance() const
{
    double variance;
    get < double > (id_(), "variance",variance);
    return variance;
}
void NormalRng::__set_variance( double variance )
{
    set < double > (id_(), "variance", variance);
}
int NormalRng::__get_method() const
{
    int method;
    get < int > (id_(), "method",method);
    return method;
}
void NormalRng::__set_method( int method )
{
    set < int > (id_(), "method", method);
}
#endif
