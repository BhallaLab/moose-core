#ifndef _pymoose_ExponentialRng_cpp
#define _pymoose_ExponentialRng_cpp
#include "ExponentialRng.h"
using namespace pymoose;
const std::string ExponentialRng::className = "ExponentialRng";
ExponentialRng::ExponentialRng(Id id):RandGenerator(id){}
ExponentialRng::ExponentialRng(std::string path):RandGenerator(className, path){}
ExponentialRng::ExponentialRng(std::string name, Id parentId):RandGenerator(className, name, parentId){}
ExponentialRng::ExponentialRng(std::string name, PyMooseBase& parent):RandGenerator(className, name, parent){}
ExponentialRng::~ExponentialRng(){}
const std::string& ExponentialRng::getType(){ return className; }
double ExponentialRng::__get_mean() const
{
    double mean;
    get < double > (id_(), "mean",mean);
    return mean;
}
void ExponentialRng::__set_mean( double mean )
{
    set < double > (id_(), "mean", mean);
}
int ExponentialRng::__get_method() const
{
    int method;
    get < int > (id_(), "method",method);
    return method;
}
void ExponentialRng::__set_method( int method )
{
    set < int > (id_(), "method", method);
}
#endif
