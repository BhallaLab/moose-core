#ifndef _pymoose_NMDAChan_cpp
#define _pymoose_NMDAChan_cpp
#include "NMDAChan.h"
using namespace pymoose;
const std::string NMDAChan::className_ = "NMDAChan";
NMDAChan::NMDAChan(Id id):SynChan(id){}
NMDAChan::NMDAChan(std::string path):SynChan(className_, path){}
NMDAChan::NMDAChan(std::string name, Id parentId):SynChan(className_, name, parentId){}
NMDAChan::NMDAChan(std::string name, PyMooseBase& parent):SynChan(className_, name, parent){}
NMDAChan::NMDAChan(const NMDAChan& src, std::string objectName, PyMooseBase& parent):SynChan(src, objectName, parent){}
NMDAChan::NMDAChan(const NMDAChan& src, std::string objectName, Id& parent):SynChan(src, objectName, parent){}
NMDAChan::NMDAChan(const NMDAChan& src, std::string path):SynChan(src, path){}
NMDAChan::NMDAChan(const Id& src, std::string name, Id& parent):SynChan(src, name, parent){}
NMDAChan::NMDAChan(const Id& src, std::string path):SynChan(src, path){}
NMDAChan::~NMDAChan(){}
const std::string& NMDAChan::getType(){ return className_; }
double NMDAChan::getTransitionParam(const unsigned int index) const
{
    double transitionParam;
    lookupGet < double, unsigned int > (id_(), "transitionParam",transitionParam, index);
    return transitionParam;
}
void NMDAChan::setTransitionParam( const unsigned int index,double transitionParam )
{
    lookupSet < double, unsigned int > (id_(), "transitionParam", transitionParam, index);
}
double NMDAChan::__get_MgConc() const
{
    double MgConc;
    get < double > (id_(), "MgConc",MgConc);
    return MgConc;
}
void NMDAChan::__set_MgConc( double MgConc )
{
    set < double > (id_(), "MgConc", MgConc);
}
double NMDAChan::__get_unblocked() const
{
    double unblocked;
    get < double > (id_(), "unblocked",unblocked);
    return unblocked;
}
double NMDAChan::__get_saturation() const
{
    double saturation;
    get < double > (id_(), "saturation",saturation);
    return saturation;
}
void NMDAChan::__set_saturation( double saturation )
{
    set < double > (id_(), "saturation", saturation);
}
#endif
