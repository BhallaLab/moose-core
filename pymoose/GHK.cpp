#ifndef _pymoose_GHK_cpp
#define _pymoose_GHK_cpp
#include "GHK.h"
using namespace pymoose;
const std::string GHK::className_ = "GHK";
GHK::GHK(Id id):Neutral(id){}
GHK::GHK(std::string path):Neutral(className_, path){}
GHK::GHK(std::string name, Id parentId):Neutral(className_, name, parentId){}
GHK::GHK(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
GHK::GHK(const GHK& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
GHK::GHK(const GHK& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
GHK::GHK(const GHK& src, std::string path):Neutral(src, path){}
GHK::GHK(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
GHK::GHK(const Id& src, std::string path):Neutral(src, path){}
GHK::~GHK(){}
const std::string& GHK::getType(){ return className_; }
double GHK::__get_Ik() const
{
    double Ik;
    get < double > (id_(), "Ik",Ik);
    return Ik;
}
double GHK::__get_Gk() const
{
    double Gk;
    get < double > (id_(), "Gk",Gk);
    return Gk;
}
double GHK::__get_Ek() const
{
    double Ek;
    get < double > (id_(), "Ek",Ek);
    return Ek;
}
double GHK::__get_T() const
{
    double T;
    get < double > (id_(), "T",T);
    return T;
}
void GHK::__set_T( double T )
{
    set < double > (id_(), "T", T);
}
double GHK::__get_p() const
{
    double p;
    get < double > (id_(), "p",p);
    return p;
}
void GHK::__set_p( double p )
{
    set < double > (id_(), "p", p);
}
double GHK::__get_Vm() const
{
    double Vm;
    get < double > (id_(), "Vm",Vm);
    return Vm;
}
void GHK::__set_Vm( double Vm )
{
    set < double > (id_(), "Vm", Vm);
}
double GHK::__get_Cin() const
{
    double Cin;
    get < double > (id_(), "Cin",Cin);
    return Cin;
}
void GHK::__set_Cin( double Cin )
{
    set < double > (id_(), "Cin", Cin);
}
double GHK::__get_Cout() const
{
    double Cout;
    get < double > (id_(), "Cout",Cout);
    return Cout;
}
void GHK::__set_Cout( double Cout )
{
    set < double > (id_(), "Cout", Cout);
}
double GHK::__get_valency() const
{
    double valency;
    get < double > (id_(), "valency",valency);
    return valency;
}
void GHK::__set_valency( double valency )
{
    set < double > (id_(), "valency", valency);
}
#endif
