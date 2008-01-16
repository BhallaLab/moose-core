#ifndef _pymoose_GammaRng_cpp
#define _pymoose_GammaRng_cpp
#include "GammaRng.h"
using namespace pymoose;
const std::string GammaRng::className = "GammaRng";
GammaRng::GammaRng(Id id):RandGenerator(id){}
GammaRng::GammaRng(std::string path):RandGenerator(className, path){}
GammaRng::GammaRng(std::string name, Id parentId):RandGenerator(className, name, parentId){}
GammaRng::GammaRng(std::string name, PyMooseBase* parent):RandGenerator(className, name, parent){}
GammaRng::~GammaRng(){}
const std::string& GammaRng::getType(){ return className; }
double GammaRng::__get_alpha() const
{
    double alpha;
    get < double > (id_(), "alpha",alpha);
    return alpha;
}
void GammaRng::__set_alpha( double alpha )
{
    set < double > (id_(), "alpha", alpha);
}
double GammaRng::__get_theta() const
{
    double theta;
    get < double > (id_(), "theta",theta);
    return theta;
}
void GammaRng::__set_theta( double theta )
{
    set < double > (id_(), "theta", theta);
}
#endif
