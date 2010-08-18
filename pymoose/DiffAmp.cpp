#ifndef _pymoose_DiffAmp_cpp
#define _pymoose_DiffAmp_cpp
#include "DiffAmp.h"
using namespace pymoose;
const std::string DiffAmp::className_ = "DiffAmp";
DiffAmp::DiffAmp(Id id):Neutral(id){}
DiffAmp::DiffAmp(std::string path):Neutral(className_, path){}
DiffAmp::DiffAmp(std::string name, Id parentId):Neutral(className_, name, parentId){}
DiffAmp::DiffAmp(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
DiffAmp::DiffAmp(const DiffAmp& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
DiffAmp::DiffAmp(const DiffAmp& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
DiffAmp::DiffAmp(const DiffAmp& src, std::string path):Neutral(src, path){}
DiffAmp::DiffAmp(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
DiffAmp::DiffAmp(const Id& src, std::string path):Neutral(src, path){}
DiffAmp::~DiffAmp(){}
const std::string& DiffAmp::getType(){ return className_; }
double DiffAmp::__get_gain() const
{
    double gain;
    get < double > (id_(), "gain",gain);
    return gain;
}
void DiffAmp::__set_gain( double gain )
{
    set < double > (id_(), "gain", gain);
}
double DiffAmp::__get_saturation() const
{
    double saturation;
    get < double > (id_(), "saturation",saturation);
    return saturation;
}
void DiffAmp::__set_saturation( double saturation )
{
    set < double > (id_(), "saturation", saturation);
}
double DiffAmp::__get_plus() const
{
    double plus;
    get < double > (id_(), "plus",plus);
    return plus;
}
double DiffAmp::__get_minus() const
{
    double minus;
    get < double > (id_(), "minus",minus);
    return minus;
}
double DiffAmp::__get_output() const
{
    double output;
    get < double > (id_(), "output",output);
    return output;
}
#endif
