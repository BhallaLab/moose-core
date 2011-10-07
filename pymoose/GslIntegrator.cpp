#ifndef _pymoose_GslIntegrator_cpp
#define _pymoose_GslIntegrator_cpp
#include "GslIntegrator.h"
using namespace pymoose;
const std::string GslIntegrator::className_ = "GslIntegrator";
GslIntegrator::GslIntegrator(std::string className, std::string objectName, Id parentId):Neutral(className, objectName, parentId){}
GslIntegrator::GslIntegrator(std::string className, std::string path):Neutral(className, path){}
GslIntegrator::GslIntegrator(std::string className, std::string objectName, PyMooseBase& parent):Neutral(className, objectName, parent){}
GslIntegrator::GslIntegrator(Id id):Neutral(id){}
GslIntegrator::GslIntegrator(std::string path):Neutral(className_, path){}
GslIntegrator::GslIntegrator(std::string name, Id parentId):Neutral(className_, name, parentId){}
GslIntegrator::GslIntegrator(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
GslIntegrator::GslIntegrator(const GslIntegrator& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
GslIntegrator::GslIntegrator(const GslIntegrator& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
GslIntegrator::GslIntegrator(const GslIntegrator& src, std::string path):Neutral(src, path){}
GslIntegrator::GslIntegrator(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
GslIntegrator::GslIntegrator(const Id& src, std::string path):Neutral(src, path){}
GslIntegrator::~GslIntegrator(){}
const std::string& GslIntegrator::getType(){ return className_; }
bool GslIntegrator::__get_isInitiatilized() const
{
    bool isInitiatilized;
    get < bool > (id_(), "isInitiatilized",isInitiatilized);
    return isInitiatilized;
}
const string&  GslIntegrator::__get_method() const
{
return this->getField("method");
}
void GslIntegrator::__set_method( string method )
{
    set < string > (id_(), "method", method);
}
double GslIntegrator::__get_relativeAccuracy() const
{
    double relativeAccuracy;
    get < double > (id_(), "relativeAccuracy",relativeAccuracy);
    return relativeAccuracy;
}
void GslIntegrator::__set_relativeAccuracy( double relativeAccuracy )
{
    set < double > (id_(), "relativeAccuracy", relativeAccuracy);
}
double GslIntegrator::__get_absoluteAccuracy() const
{
    double absoluteAccuracy;
    get < double > (id_(), "absoluteAccuracy",absoluteAccuracy);
    return absoluteAccuracy;
}
void GslIntegrator::__set_absoluteAccuracy( double absoluteAccuracy )
{
    set < double > (id_(), "absoluteAccuracy", absoluteAccuracy);
}
double GslIntegrator::__get_internalDt() const
{
    double internalDt;
    get < double > (id_(), "internalDt",internalDt);
    return internalDt;
}
void GslIntegrator::__set_internalDt( double internalDt )
{
    set < double > (id_(), "internalDt", internalDt);
}
#endif
