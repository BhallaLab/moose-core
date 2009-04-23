#ifndef _pymoose_HHChannel_cpp
#define _pymoose_HHChannel_cpp
#include "HHChannel.h"
using namespace pymoose;
const std::string HHChannel::className_ = "HHChannel";
HHChannel::HHChannel(Id id):PyMooseBase(id){}
HHChannel::HHChannel(std::string path):PyMooseBase(className_, path){}
HHChannel::HHChannel(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
HHChannel::HHChannel(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
HHChannel::HHChannel(const HHChannel& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
HHChannel::HHChannel(const HHChannel& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
HHChannel::HHChannel(const HHChannel& src, std::string path):PyMooseBase(src, path){}
HHChannel::HHChannel(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
HHChannel::~HHChannel(){}
const std::string& HHChannel::getType(){ return className_; }
double HHChannel::__get_Gbar() const
{
    double Gbar;
    get < double > (id_(), "Gbar",Gbar);
    return Gbar;
}
void HHChannel::__set_Gbar( double Gbar )
{
    set < double > (id_(), "Gbar", Gbar);
}
double HHChannel::__get_Ek() const
{
    double Ek;
    get < double > (id_(), "Ek",Ek);
    return Ek;
}
void HHChannel::__set_Ek( double Ek )
{
    set < double > (id_(), "Ek", Ek);
}
double HHChannel::__get_Xpower() const
{
    double Xpower;
    get < double > (id_(), "Xpower",Xpower);
    return Xpower;
}
void HHChannel::__set_Xpower( double Xpower )
{
    set < double > (id_(), "Xpower", Xpower);
}
double HHChannel::__get_Ypower() const
{
    double Ypower;
    get < double > (id_(), "Ypower",Ypower);
    return Ypower;
}
void HHChannel::__set_Ypower( double Ypower )
{
    set < double > (id_(), "Ypower", Ypower);
}
double HHChannel::__get_Zpower() const
{
    double Zpower;
    get < double > (id_(), "Zpower",Zpower);
    return Zpower;
}
void HHChannel::__set_Zpower( double Zpower )
{
    set < double > (id_(), "Zpower", Zpower);
}
int HHChannel::__get_instant() const
{
    int instant;
    get < int > (id_(), "instant",instant);
    return instant;
}
void HHChannel::__set_instant( int instant )
{
    set < int > (id_(), "instant", instant);
}
double HHChannel::__get_Gk() const
{
    double Gk;
    get < double > (id_(), "Gk",Gk);
    return Gk;
}
void HHChannel::__set_Gk( double Gk )
{
    set < double > (id_(), "Gk", Gk);
}
double HHChannel::__get_Ik() const
{
    double Ik;
    get < double > (id_(), "Ik",Ik);
    return Ik;
}
double HHChannel::__get_X() const
{
    double X;
    get < double > (id_(), "X",X);
    return X;
}
void HHChannel::__set_X( double X )
{
    set < double > (id_(), "X", X);
}
double HHChannel::__get_Y() const
{
    double Y;
    get < double > (id_(), "Y",Y);
    return Y;
}
void HHChannel::__set_Y( double Y )
{
    set < double > (id_(), "Y", Y);
}
double HHChannel::__get_Z() const
{
    double Z;
    get < double > (id_(), "Z",Z);
    return Z;
}
void HHChannel::__set_Z( double Z )
{
    set < double > (id_(), "Z", Z);
}
int HHChannel::__get_useConcentration() const
{
    int useConcentration;
    get < int > (id_(), "useConcentration",useConcentration);
    return useConcentration;
}
void HHChannel::__set_useConcentration( int useConcentration )
{
    set < int > (id_(), "useConcentration", useConcentration);
}
#endif
