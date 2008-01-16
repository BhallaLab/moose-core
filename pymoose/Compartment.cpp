#ifndef _pymoose_Compartment_cpp
#define _pymoose_Compartment_cpp
#include "Compartment.h"
using namespace pymoose;

const std::string Compartment::className = "Compartment";
Compartment::Compartment(Id id):PyMooseBase(id){}
Compartment::Compartment(std::string path):PyMooseBase(className, path){}
Compartment::Compartment(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Compartment::Compartment(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Compartment::~Compartment(){}
const std::string& Compartment::getType(){ return className; }
double Compartment::__get_Vm() const
{
    double Vm;
    get < double > (id_(), "Vm",Vm);
    return Vm;
}
void Compartment::__set_Vm( double Vm )
{
    set < double > (id_(), "Vm", Vm);
}
double Compartment::__get_Cm() const
{
    double Cm;
    get < double > (id_(), "Cm",Cm);
    return Cm;
}
void Compartment::__set_Cm( double Cm )
{
    set < double > (id_(), "Cm", Cm);
}
double Compartment::__get_Em() const
{
    double Em;
    get < double > (id_(), "Em",Em);
    return Em;
}
void Compartment::__set_Em( double Em )
{
    set < double > (id_(), "Em", Em);
}
double Compartment::__get_Im() const
{
    double Im;
    get < double > (id_(), "Im",Im);
    return Im;
}
void Compartment::__set_Im( double Im )
{
    set < double > (id_(), "Im", Im);
}
double Compartment::__get_inject() const
{
    double inject;
    get < double > (id_(), "inject",inject);
    return inject;
}
void Compartment::__set_inject( double inject )
{
    set < double > (id_(), "inject", inject);
}
double Compartment::__get_initVm() const
{
    double initVm;
    get < double > (id_(), "initVm",initVm);
    return initVm;
}
void Compartment::__set_initVm( double initVm )
{
    set < double > (id_(), "initVm", initVm);
}
double Compartment::__get_Rm() const
{
    double Rm;
    get < double > (id_(), "Rm",Rm);
    return Rm;
}
void Compartment::__set_Rm( double Rm )
{
    set < double > (id_(), "Rm", Rm);
}
double Compartment::__get_Ra() const
{
    double Ra;
    get < double > (id_(), "Ra",Ra);
    return Ra;
}
void Compartment::__set_Ra( double Ra )
{
    set < double > (id_(), "Ra", Ra);
}
double Compartment::__get_diameter() const
{
    double diameter;
    get < double > (id_(), "diameter",diameter);
    return diameter;
}
void Compartment::__set_diameter( double diameter )
{
    set < double > (id_(), "diameter", diameter);
}
double Compartment::__get_length() const
{
    double length;
    get < double > (id_(), "length",length);
    return length;
}
void Compartment::__set_length( double length )
{
    set < double > (id_(), "length", length);
}
double Compartment::__get_x() const
{
    double x;
    get < double > (id_(), "x",x);
    return x;
}
void Compartment::__set_x( double x )
{
    set < double > (id_(), "x", x);
}
double Compartment::__get_y() const
{
    double y;
    get < double > (id_(), "y",y);
    return y;
}
void Compartment::__set_y( double y )
{
    set < double > (id_(), "y", y);
}
double Compartment::__get_z() const
{
    double z;
    get < double > (id_(), "z",z);
    return z;
}
void Compartment::__set_z( double z )
{
    set < double > (id_(), "z", z);
}
double Compartment::__get_VmSrc() const
{
    double VmSrc;
    get < double > (id_(), "VmSrc",VmSrc);
    return VmSrc;
}
void Compartment::__set_VmSrc( double VmSrc )
{
    set < double > (id_(), "VmSrc", VmSrc);
}
double Compartment::__get_injectMsg() const
{
    double injectMsg;
    get < double > (id_(), "injectMsg",injectMsg);
    return injectMsg;
}
void Compartment::__set_injectMsg( double injectMsg )
{
    set < double > (id_(), "injectMsg", injectMsg);
}
#endif
