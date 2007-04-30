#ifndef _pymoose_SpikeGen_cpp
#define _pymoose_SpikeGen_cpp
#include "SpikeGen.h"
const std::string SpikeGen::className = "SpikeGen";
SpikeGen::SpikeGen(Id id):PyMooseBase(id){}
SpikeGen::SpikeGen(std::string path):PyMooseBase(className, path){}
SpikeGen::SpikeGen(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
SpikeGen::SpikeGen(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
SpikeGen::~SpikeGen(){}
const std::string& SpikeGen::getType(){ return className; }
double SpikeGen::__get_threshold() const
{
    double threshold;
    get < double > (Element::element(id_), "threshold",threshold);
    return threshold;
}
void SpikeGen::__set_threshold( double threshold )
{
    set < double > (Element::element(id_), "threshold", threshold);
}
double SpikeGen::__get_refractT() const
{
    double refractT;
    get < double > (Element::element(id_), "refractT",refractT);
    return refractT;
}
void SpikeGen::__set_refractT( double refractT )
{
    set < double > (Element::element(id_), "refractT", refractT);
}
double SpikeGen::__get_abs_refract() const
{
    double abs_refract;
    get < double > (Element::element(id_), "abs_refract",abs_refract);
    return abs_refract;
}
void SpikeGen::__set_abs_refract( double abs_refract )
{
    set < double > (Element::element(id_), "abs_refract", abs_refract);
}
double SpikeGen::__get_amplitude() const
{
    double amplitude;
    get < double > (Element::element(id_), "amplitude",amplitude);
    return amplitude;
}
void SpikeGen::__set_amplitude( double amplitude )
{
    set < double > (Element::element(id_), "amplitude", amplitude);
}
double SpikeGen::__get_state() const
{
    double state;
    get < double > (Element::element(id_), "state",state);
    return state;
}
void SpikeGen::__set_state( double state )
{
    set < double > (Element::element(id_), "state", state);
}
double SpikeGen::__get_event() const
{
    double event;
    get < double > (Element::element(id_), "event",event);
    return event;
}
void SpikeGen::__set_event( double event )
{
    set < double > (Element::element(id_), "event", event);
}
double SpikeGen::__get_Vm() const
{
    double Vm;
    get < double > (Element::element(id_), "Vm",Vm);
    return Vm;
}
void SpikeGen::__set_Vm( double Vm )
{
    set < double > (Element::element(id_), "Vm", Vm);
}
#endif
