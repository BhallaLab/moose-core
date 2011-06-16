#ifndef _pymoose_STPNMDAChan_cpp
#define _pymoose_STPNMDAChan_cpp
#include "STPNMDAChan.h"
using namespace pymoose;
const std::string STPNMDAChan::className_ = "STPNMDAChan";
STPNMDAChan::STPNMDAChan(std::string className, std::string objectName, Id parentId):STPSynChan(className, objectName, parentId){}
STPNMDAChan::STPNMDAChan(std::string className, std::string path):STPSynChan(className, path){}
STPNMDAChan::STPNMDAChan(std::string className, std::string objectName, PyMooseBase& parent):STPSynChan(className, objectName, parent){}
STPNMDAChan::STPNMDAChan(Id id):STPSynChan(id){}
STPNMDAChan::STPNMDAChan(std::string path):STPSynChan(className_, path){}
STPNMDAChan::STPNMDAChan(std::string name, Id parentId):STPSynChan(className_, name, parentId){}
STPNMDAChan::STPNMDAChan(std::string name, PyMooseBase& parent):STPSynChan(className_, name, parent){}
STPNMDAChan::STPNMDAChan(const STPNMDAChan& src, std::string objectName, PyMooseBase& parent):STPSynChan(src, objectName, parent){}
STPNMDAChan::STPNMDAChan(const STPNMDAChan& src, std::string objectName, Id& parent):STPSynChan(src, objectName, parent){}
STPNMDAChan::STPNMDAChan(const STPNMDAChan& src, std::string path):STPSynChan(src, path){}
STPNMDAChan::STPNMDAChan(const Id& src, std::string name, Id& parent):STPSynChan(src, name, parent){}
STPNMDAChan::STPNMDAChan(const Id& src, std::string path):STPSynChan(src, path){}
STPNMDAChan::~STPNMDAChan(){}
const std::string& STPNMDAChan::getType(){ return className_; }
double STPNMDAChan::__get_MgConc() const
{
    double MgConc;
    get < double > (id_(), "MgConc",MgConc);
    return MgConc;
}
void STPNMDAChan::__set_MgConc( double MgConc )
{
    set < double > (id_(), "MgConc", MgConc);
}
double STPNMDAChan::__get_unblocked() const
{
    double unblocked;
    get < double > (id_(), "unblocked",unblocked);
    return unblocked;
}
double STPNMDAChan::__get_saturation() const
{
    double saturation;
    get < double > (id_(), "saturation",saturation);
    return saturation;
}
void STPNMDAChan::__set_saturation( double saturation )
{
    set < double > (id_(), "saturation", saturation);
}

// The following were added manually
double STPNMDAChan::getTransitionParam(const unsigned int index) const
{
    double transitionParam;
    lookupGet < double, unsigned int > (id_(), "transitionParam",transitionParam, index);
    return transitionParam;
}
void STPNMDAChan::setTransitionParam( const unsigned int index,double transitionParam )
{
    lookupSet < double, unsigned int > (id_(), "transitionParam", transitionParam, index);
}

#endif
