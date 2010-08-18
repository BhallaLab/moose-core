#ifndef _pymoose_IntFire_cpp
#define _pymoose_IntFire_cpp
#include "IntFire.h"
using namespace pymoose;
const std::string IntFire::className_ = "IntFire";
IntFire::IntFire(Id id):Neutral(id){}
IntFire::IntFire(std::string path):Neutral(className_, path){}
IntFire::IntFire(std::string name, Id parentId):Neutral(className_, name, parentId){}
IntFire::IntFire(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
IntFire::IntFire(const IntFire& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
IntFire::IntFire(const IntFire& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
IntFire::IntFire(const IntFire& src, std::string path):Neutral(src, path){}
IntFire::IntFire(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
IntFire::IntFire(const Id& src, std::string path):Neutral(src, path){}
IntFire::~IntFire(){}
const std::string& IntFire::getType(){ return className_; }
double IntFire::__get_Vt() const
{
    double Vt;
    get < double > (id_(), "Vt",Vt);
    return Vt;
}
void IntFire::__set_Vt( double Vt )
{
    set < double > (id_(), "Vt", Vt);
}
double IntFire::__get_Vr() const
{
    double Vr;
    get < double > (id_(), "Vr",Vr);
    return Vr;
}
void IntFire::__set_Vr( double Vr )
{
    set < double > (id_(), "Vr", Vr);
}
double IntFire::__get_Rm() const
{
    double Rm;
    get < double > (id_(), "Rm",Rm);
    return Rm;
}
void IntFire::__set_Rm( double Rm )
{
    set < double > (id_(), "Rm", Rm);
}
double IntFire::__get_Cm() const
{
    double Cm;
    get < double > (id_(), "Cm",Cm);
    return Cm;
}
void IntFire::__set_Cm( double Cm )
{
    set < double > (id_(), "Cm", Cm);
}
double IntFire::__get_Vm() const
{
    double Vm;
    get < double > (id_(), "Vm",Vm);
    return Vm;
}
void IntFire::__set_Vm( double Vm )
{
    set < double > (id_(), "Vm", Vm);
}
double IntFire::__get_tau() const
{
    double tau;
    get < double > (id_(), "tau",tau);
    return tau;
}
double IntFire::__get_Em() const
{
    double Em;
    get < double > (id_(), "Em",Em);
    return Em;
}
void IntFire::__set_Em( double Em )
{
    set < double > (id_(), "Em", Em);
}
double IntFire::__get_refractT() const
{
    double refractT;
    get < double > (id_(), "refractT",refractT);
    return refractT;
}
void IntFire::__set_refractT( double refractT )
{
    set < double > (id_(), "refractT", refractT);
}
double IntFire::__get_initVm() const
{
    double initVm;
    get < double > (id_(), "initVm",initVm);
    return initVm;
}
void IntFire::__set_initVm( double initVm )
{
    set < double > (id_(), "initVm", initVm);
}
double IntFire::__get_inject() const
{
    double inject;
    get < double > (id_(), "inject",inject);
    return inject;
}
void IntFire::__set_inject( double inject )
{
    set < double > (id_(), "inject", inject);
}
#endif
