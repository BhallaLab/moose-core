#ifndef _pymoose_HHChannel_cpp
#define _pymoose_HHChannel_cpp
#include "HHChannel.h"
const std::string HHChannel::className = "HHChannel";
HHChannel::HHChannel(Id id):PyMooseBase(id){}
HHChannel::HHChannel(std::string path):PyMooseBase(className, path){}
HHChannel::HHChannel(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
HHChannel::HHChannel(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
HHChannel::~HHChannel(){}
const std::string& HHChannel::getType(){ return className; }
double HHChannel::__get_Gbar() const
{
    double Gbar;
    get < double > (Element::element(id_), "Gbar",Gbar);
    return Gbar;
}
void HHChannel::__set_Gbar( double Gbar )
{
    set < double > (Element::element(id_), "Gbar", Gbar);
}
double HHChannel::__get_Ek() const
{
    double Ek;
    get < double > (Element::element(id_), "Ek",Ek);
    return Ek;
}
void HHChannel::__set_Ek( double Ek )
{
    set < double > (Element::element(id_), "Ek", Ek);
}
double HHChannel::__get_Xpower() const
{
    double Xpower;
    get < double > (Element::element(id_), "Xpower",Xpower);
    return Xpower;
}
void HHChannel::__set_Xpower( double Xpower )
{
    set < double > (Element::element(id_), "Xpower", Xpower);
}
double HHChannel::__get_Ypower() const
{
    double Ypower;
    get < double > (Element::element(id_), "Ypower",Ypower);
    return Ypower;
}
void HHChannel::__set_Ypower( double Ypower )
{
    set < double > (Element::element(id_), "Ypower", Ypower);
}
double HHChannel::__get_Zpower() const
{
    double Zpower;
    get < double > (Element::element(id_), "Zpower",Zpower);
    return Zpower;
}
void HHChannel::__set_Zpower( double Zpower )
{
    set < double > (Element::element(id_), "Zpower", Zpower);
}
int HHChannel::__get_instant() const
{
    int instant;
    get < int > (Element::element(id_), "instant",instant);
    return instant;
}
void HHChannel::__set_instant( int instant )
{
    set < int > (Element::element(id_), "instant", instant);
}
double HHChannel::__get_Gk() const
{
    double Gk;
    get < double > (Element::element(id_), "Gk",Gk);
    return Gk;
}
void HHChannel::__set_Gk( double Gk )
{
    set < double > (Element::element(id_), "Gk", Gk);
}
double HHChannel::__get_Ik() const
{
    double Ik;
    get < double > (Element::element(id_), "Ik",Ik);
    return Ik;
}
void HHChannel::__set_Ik( double Ik )
{
    set < double > (Element::element(id_), "Ik", Ik);
}
int HHChannel::__get_useConcentration() const
{
    int useConcentration;
    get < int > (Element::element(id_), "useConcentration",useConcentration);
    return useConcentration;
}
void HHChannel::__set_useConcentration( int useConcentration )
{
    set < int > (Element::element(id_), "useConcentration", useConcentration);
}
double HHChannel::__get_IkSrc() const
{
    double IkSrc;
    get < double > (Element::element(id_), "IkSrc",IkSrc);
    return IkSrc;
}
void HHChannel::__set_IkSrc( double IkSrc )
{
    set < double > (Element::element(id_), "IkSrc", IkSrc);
}
double HHChannel::__get_concen() const
{
    double concen;
    get < double > (Element::element(id_), "concen",concen);
    return concen;
}
void HHChannel::__set_concen( double concen )
{
    set < double > (Element::element(id_), "concen", concen);
}
#endif
