#ifndef _pymoose_KinSynChan_cpp
#define _pymoose_KinSynChan_cpp
#include "KinSynChan.h"
using namespace pymoose;
const std::string KinSynChan::className_ = "KinSynChan";
KinSynChan::KinSynChan(Id id):SynChan(id){}
KinSynChan::KinSynChan(std::string path):SynChan(className_, path){}
KinSynChan::KinSynChan(std::string name, Id parentId):SynChan(className_, name, parentId){}
KinSynChan::KinSynChan(std::string name, PyMooseBase& parent):SynChan(className_, name, parent){}
KinSynChan::KinSynChan(const KinSynChan& src, std::string objectName, PyMooseBase& parent):SynChan(src, objectName, parent){}
KinSynChan::KinSynChan(const KinSynChan& src, std::string objectName, Id& parent):SynChan(src, objectName, parent){}
KinSynChan::KinSynChan(const KinSynChan& src, std::string path):SynChan(src, path){}
KinSynChan::KinSynChan(const Id& src, std::string name, Id& parent):SynChan(src, name, parent){}
KinSynChan::KinSynChan(const Id& src, std::string path):SynChan(src, path){}
KinSynChan::~KinSynChan(){}
const std::string& KinSynChan::getType(){ return className_; }
double KinSynChan::__get_rInf() const
{
    double rInf;
    get < double > (id_(), "rInf",rInf);
    return rInf;
}
void KinSynChan::__set_rInf( double rInf )
{
    set < double > (id_(), "rInf", rInf);
}
double KinSynChan::__get_tauR() const
{
    double tauR;
    get < double > (id_(), "tauR",tauR);
    return tauR;
}
void KinSynChan::__set_tauR( double tauR )
{
    set < double > (id_(), "tauR", tauR);
}
double KinSynChan::__get_pulseWidth() const
{
    double pulseWidth;
    get < double > (id_(), "pulseWidth",pulseWidth);
    return pulseWidth;
}
void KinSynChan::__set_pulseWidth( double pulseWidth )
{
    set < double > (id_(), "pulseWidth", pulseWidth);
}
#endif
