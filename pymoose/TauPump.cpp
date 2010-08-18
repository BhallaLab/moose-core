#ifndef _pymoose_TauPump_cpp
#define _pymoose_TauPump_cpp
#include "TauPump.h"
using namespace pymoose;
const std::string TauPump::className_ = "TauPump";
TauPump::TauPump(Id id):Neutral(id){}
TauPump::TauPump(std::string path):Neutral(className_, path){}
TauPump::TauPump(std::string name, Id parentId):Neutral(className_, name, parentId){}
TauPump::TauPump(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
TauPump::TauPump(const TauPump& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
TauPump::TauPump(const TauPump& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
TauPump::TauPump(const TauPump& src, std::string path):Neutral(src, path){}
TauPump::TauPump(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
TauPump::TauPump(const Id& src, std::string path):Neutral(src, path){}
TauPump::~TauPump(){}
const std::string& TauPump::getType(){ return className_; }
double TauPump::__get_pumpRate() const
{
    double pumpRate;
    get < double > (id_(), "pumpRate",pumpRate);
    return pumpRate;
}
void TauPump::__set_pumpRate( double pumpRate )
{
    set < double > (id_(), "pumpRate", pumpRate);
}
double TauPump::__get_eqConc() const
{
    double eqConc;
    get < double > (id_(), "eqConc",eqConc);
    return eqConc;
}
void TauPump::__set_eqConc( double eqConc )
{
    set < double > (id_(), "eqConc", eqConc);
}
double TauPump::__get_TA() const
{
    double TA;
    get < double > (id_(), "TA",TA);
    return TA;
}
void TauPump::__set_TA( double TA )
{
    set < double > (id_(), "TA", TA);
}
double TauPump::__get_TB() const
{
    double TB;
    get < double > (id_(), "TB",TB);
    return TB;
}
void TauPump::__set_TB( double TB )
{
    set < double > (id_(), "TB", TB);
}
double TauPump::__get_TC() const
{
    double TC;
    get < double > (id_(), "TC",TC);
    return TC;
}
void TauPump::__set_TC( double TC )
{
    set < double > (id_(), "TC", TC);
}
double TauPump::__get_TV() const
{
    double TV;
    get < double > (id_(), "TV",TV);
    return TV;
}
void TauPump::__set_TV( double TV )
{
    set < double > (id_(), "TV", TV);
}
#endif
