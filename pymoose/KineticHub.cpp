#ifndef _pymoose_KineticHub_cpp
#define _pymoose_KineticHub_cpp
#include "KineticHub.h"
using namespace pymoose;
const std::string KineticHub::className_ = "KineticHub";
KineticHub::KineticHub(Id id):Neutral(id){}
KineticHub::KineticHub(std::string path):Neutral(className_, path){}
KineticHub::KineticHub(std::string name, Id parentId):Neutral(className_, name, parentId){}
KineticHub::KineticHub(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
KineticHub::KineticHub(const KineticHub& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
KineticHub::KineticHub(const KineticHub& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
KineticHub::KineticHub(const KineticHub& src, std::string path):Neutral(src, path){}
KineticHub::KineticHub(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
KineticHub::KineticHub(const Id& src, std::string path):Neutral(src, path){}
KineticHub::~KineticHub(){}
const std::string& KineticHub::getType(){ return className_; }
unsigned int KineticHub::__get_nVarMol() const
{
    unsigned int nVarMol;
    get < unsigned int > (id_(), "nVarMol",nVarMol);
    return nVarMol;
}
unsigned int KineticHub::__get_nReac() const
{
    unsigned int nReac;
    get < unsigned int > (id_(), "nReac",nReac);
    return nReac;
}
unsigned int KineticHub::__get_nEnz() const
{
    unsigned int nEnz;
    get < unsigned int > (id_(), "nEnz",nEnz);
    return nEnz;
}
bool KineticHub::__get_zombifySeparate() const
{
    bool zombifySeparate;
    get < bool > (id_(), "zombifySeparate",zombifySeparate);
    return zombifySeparate;
}
void KineticHub::__set_zombifySeparate( bool zombifySeparate )
{
    set < bool > (id_(), "zombifySeparate", zombifySeparate);
}
#endif
