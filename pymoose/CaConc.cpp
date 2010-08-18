#ifndef _pymoose_CaConc_cpp
#define _pymoose_CaConc_cpp
#include "CaConc.h"
using namespace pymoose;
const std::string CaConc::className_ = "CaConc";
CaConc::CaConc(Id id):Neutral(id){}
CaConc::CaConc(std::string path):Neutral(className_, path){}
CaConc::CaConc(std::string name, Id parentId):Neutral(className_, name, parentId){}
CaConc::CaConc(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
CaConc::CaConc(const CaConc& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
CaConc::CaConc(const CaConc& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
CaConc::CaConc(const CaConc& src, std::string path):Neutral(src, path){}
CaConc::CaConc(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
CaConc::CaConc(const Id& src, std::string path):Neutral(src, path){}
CaConc::~CaConc(){}
const std::string& CaConc::getType(){ return className_; }
double CaConc::__get_Ca() const
{
    double Ca;
    get < double > (id_(), "Ca",Ca);
    return Ca;
}
void CaConc::__set_Ca( double Ca )
{
    set < double > (id_(), "Ca", Ca);
}
double CaConc::__get_CaBasal() const
{
    double CaBasal;
    get < double > (id_(), "CaBasal",CaBasal);
    return CaBasal;
}
void CaConc::__set_CaBasal( double CaBasal )
{
    set < double > (id_(), "CaBasal", CaBasal);
}
double CaConc::__get_Ca_base() const
{
    double Ca_base;
    get < double > (id_(), "Ca_base",Ca_base);
    return Ca_base;
}
void CaConc::__set_Ca_base( double Ca_base )
{
    set < double > (id_(), "Ca_base", Ca_base);
}
double CaConc::__get_tau() const
{
    double tau;
    get < double > (id_(), "tau",tau);
    return tau;
}
void CaConc::__set_tau( double tau )
{
    set < double > (id_(), "tau", tau);
}
double CaConc::__get_B() const
{
    double B;
    get < double > (id_(), "B",B);
    return B;
}
void CaConc::__set_B( double B )
{
    set < double > (id_(), "B", B);
}
double CaConc::__get_thick() const
{
    double thick;
    get < double > (id_(), "thick",thick);
    return thick;
}
void CaConc::__set_thick( double thick )
{
    set < double > (id_(), "thick", thick);
}
double CaConc::__get_ceiling() const
{
    double ceiling;
    get < double > (id_(), "ceiling",ceiling);
    return ceiling;
}
void CaConc::__set_ceiling( double ceiling )
{
    set < double > (id_(), "ceiling", ceiling);
}
double CaConc::__get_floor() const
{
    double floor;
    get < double > (id_(), "floor",floor);
    return floor;
}
void CaConc::__set_floor( double floor )
{
    set < double > (id_(), "floor", floor);
}
#endif
