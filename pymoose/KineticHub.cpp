#ifndef _pymoose_KineticHub_cpp
#define _pymoose_KineticHub_cpp
#include "KineticHub.h"
using namespace pymoose;
const std::string KineticHub::className = "KineticHub";
KineticHub::KineticHub(Id id):PyMooseBase(id){}
KineticHub::KineticHub(std::string path):PyMooseBase(className, path){}
KineticHub::KineticHub(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
KineticHub::KineticHub(std::string name, PyMooseBase& parent):PyMooseBase(className, name, parent){}
KineticHub::~KineticHub(){}
const std::string& KineticHub::getType(){ return className; }
unsigned int KineticHub::__get_nMol() const
{
    unsigned int nMol;
    get < unsigned int > (id_(), "nMol",nMol);
    return nMol;
}
void KineticHub::__set_nMol( unsigned int nMol )
{
    set < unsigned int > (id_(), "nMol", nMol);
}
unsigned int KineticHub::__get_nReac() const
{
    unsigned int nReac;
    get < unsigned int > (id_(), "nReac",nReac);
    return nReac;
}
void KineticHub::__set_nReac( unsigned int nReac )
{
    set < unsigned int > (id_(), "nReac", nReac);
}
unsigned int KineticHub::__get_nEnz() const
{
    unsigned int nEnz;
    get < unsigned int > (id_(), "nEnz",nEnz);
    return nEnz;
}
void KineticHub::__set_nEnz( unsigned int nEnz )
{
    set < unsigned int > (id_(), "nEnz", nEnz);
}
void KineticHub::destroy()
{
    set(id_(), "destroy");
}
double KineticHub::__get_molSum() const
{
    double molSum;
    get < double > (id_(), "molSum",molSum);
    return molSum;
}
void KineticHub::__set_molSum( double molSum )
{
    set < double > (id_(), "molSum", molSum);
}
#endif
