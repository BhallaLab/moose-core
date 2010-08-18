#ifndef _pymoose_IzhikevichNrn_cpp
#define _pymoose_IzhikevichNrn_cpp
#include "IzhikevichNrn.h"
using namespace pymoose;
const std::string IzhikevichNrn::className_ = "IzhikevichNrn";
IzhikevichNrn::IzhikevichNrn(Id id):Neutral(id){}
IzhikevichNrn::IzhikevichNrn(std::string path):Neutral(className_, path){}
IzhikevichNrn::IzhikevichNrn(std::string name, Id parentId):Neutral(className_, name, parentId){}
IzhikevichNrn::IzhikevichNrn(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
IzhikevichNrn::IzhikevichNrn(const IzhikevichNrn& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
IzhikevichNrn::IzhikevichNrn(const IzhikevichNrn& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
IzhikevichNrn::IzhikevichNrn(const IzhikevichNrn& src, std::string path):Neutral(src, path){}
IzhikevichNrn::IzhikevichNrn(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
IzhikevichNrn::IzhikevichNrn(const Id& src, std::string path):Neutral(src, path){}
IzhikevichNrn::~IzhikevichNrn(){}
const std::string& IzhikevichNrn::getType(){ return className_; }
double IzhikevichNrn::__get_Vmax() const
{
    double Vmax;
    get < double > (id_(), "Vmax",Vmax);
    return Vmax;
}
void IzhikevichNrn::__set_Vmax( double Vmax )
{
    set < double > (id_(), "Vmax", Vmax);
}
double IzhikevichNrn::__get_c() const
{
    double c;
    get < double > (id_(), "c",c);
    return c;
}
void IzhikevichNrn::__set_c( double c )
{
    set < double > (id_(), "c", c);
}
double IzhikevichNrn::__get_d() const
{
    double d;
    get < double > (id_(), "d",d);
    return d;
}
void IzhikevichNrn::__set_d( double d )
{
    set < double > (id_(), "d", d);
}
double IzhikevichNrn::__get_a() const
{
    double a;
    get < double > (id_(), "a",a);
    return a;
}
void IzhikevichNrn::__set_a( double a )
{
    set < double > (id_(), "a", a);
}
double IzhikevichNrn::__get_b() const
{
    double b;
    get < double > (id_(), "b",b);
    return b;
}
void IzhikevichNrn::__set_b( double b )
{
    set < double > (id_(), "b", b);
}
double IzhikevichNrn::__get_Vm() const
{
    double Vm;
    get < double > (id_(), "Vm",Vm);
    return Vm;
}
void IzhikevichNrn::__set_Vm( double Vm )
{
    set < double > (id_(), "Vm", Vm);
}
double IzhikevichNrn::__get_u() const
{
    double u;
    get < double > (id_(), "u",u);
    return u;
}
double IzhikevichNrn::__get_Im() const
{
    double Im;
    get < double > (id_(), "Im",Im);
    return Im;
}
double IzhikevichNrn::__get_initVm() const
{
    double initVm;
    get < double > (id_(), "initVm",initVm);
    return initVm;
}
void IzhikevichNrn::__set_initVm( double initVm )
{
    set < double > (id_(), "initVm", initVm);
}
double IzhikevichNrn::__get_initU() const
{
    double initU;
    get < double > (id_(), "initU",initU);
    return initU;
}
void IzhikevichNrn::__set_initU( double initU )
{
    set < double > (id_(), "initU", initU);
}
double IzhikevichNrn::__get_alpha() const
{
    double alpha;
    get < double > (id_(), "alpha",alpha);
    return alpha;
}
void IzhikevichNrn::__set_alpha( double alpha )
{
    set < double > (id_(), "alpha", alpha);
}
double IzhikevichNrn::__get_beta() const
{
    double beta;
    get < double > (id_(), "beta",beta);
    return beta;
}
void IzhikevichNrn::__set_beta( double beta )
{
    set < double > (id_(), "beta", beta);
}
double IzhikevichNrn::__get_gamma() const
{
    double gamma;
    get < double > (id_(), "gamma",gamma);
    return gamma;
}
void IzhikevichNrn::__set_gamma( double gamma )
{
    set < double > (id_(), "gamma", gamma);
}
double IzhikevichNrn::__get_Rm() const
{
    double Rm;
    get < double > (id_(), "Rm",Rm);
    return Rm;
}
void IzhikevichNrn::__set_Rm( double Rm )
{
    set < double > (id_(), "Rm", Rm);
}
#endif
