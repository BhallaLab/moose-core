#ifndef _pymoose_SpikeGen_cpp
#define _pymoose_SpikeGen_cpp
#include "SpikeGen.h"
using namespace pymoose;
const std::string SpikeGen::className_ = "SpikeGen";
SpikeGen::SpikeGen(Id id):PyMooseBase(id){}
SpikeGen::SpikeGen(std::string path):PyMooseBase(className_, path){}
SpikeGen::SpikeGen(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
SpikeGen::SpikeGen(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
SpikeGen::SpikeGen(const SpikeGen& src, std::string objectName,  PyMooseBase& parent):PyMooseBase(src, objectName, parent){}

SpikeGen::SpikeGen(const SpikeGen& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
SpikeGen::SpikeGen(const SpikeGen& src, std::string path):PyMooseBase(src, path)
{
}

SpikeGen::SpikeGen(const Id& src, string name, Id& parent):PyMooseBase(src, name, parent)
{
}
SpikeGen::~SpikeGen(){}
const std::string& SpikeGen::getType(){ return className_; }
double SpikeGen::__get_threshold() const
{
    double threshold;
    get < double > (id_(), "threshold",threshold);
    return threshold;
}
void SpikeGen::__set_threshold( double threshold )
{
    set < double > (id_(), "threshold", threshold);
}
double SpikeGen::__get_refractT() const
{
    double refractT;
    get < double > (id_(), "refractT",refractT);
    return refractT;
}
void SpikeGen::__set_refractT( double refractT )
{
    set < double > (id_(), "refractT", refractT);
}
double SpikeGen::__get_absRefractT() const
{
    double abs_refract;
    get < double > (id_(), "abs_refract",abs_refract);
    return abs_refract;
}
void SpikeGen::__set_absRefractT( double abs_refract )
{
    set < double > (id_(), "abs_refract", abs_refract);
}
double SpikeGen::__get_amplitude() const
{
    double amplitude;
    get < double > (id_(), "amplitude",amplitude);
    return amplitude;
}
void SpikeGen::__set_amplitude( double amplitude )
{
    set < double > (id_(), "amplitude", amplitude);
}
double SpikeGen::__get_state() const
{
    double state;
    get < double > (id_(), "state",state);
    return state;
}
void SpikeGen::__set_state( double state )
{
    set < double > (id_(), "state", state);
}
int SpikeGen::__get_edgeTriggered() const
{
    int edgeTriggered;
    get < int > (id_(), "edgeTriggered",edgeTriggered);
    return edgeTriggered;
}
void SpikeGen::__set_edgeTriggered( int edgeTriggered )
{
    set < int > (id_(), "edgeTriggered", edgeTriggered);
}

#endif
