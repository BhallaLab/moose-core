#ifndef _pymoose_CaConc_cpp
#define _pymoose_CaConc_cpp
#include "CaConc.h"
const std::string CaConc::className = "CaConc";
CaConc::CaConc(Id id):PyMooseBase(id){}
CaConc::CaConc(std::string path):PyMooseBase(className, path){}
CaConc::CaConc(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
CaConc::CaConc(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
CaConc::~CaConc(){}
const std::string& CaConc::getType(){ return className; }
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
double CaConc::__get_concSrc() const
{
    double concSrc;
    get < double > (id_(), "concSrc",concSrc);
    return concSrc;
}
void CaConc::__set_concSrc( double concSrc )
{
    set < double > (id_(), "concSrc", concSrc);
}
double CaConc::__get_current() const
{
    double current;
    get < double > (id_(), "current",current);
    return current;
}
void CaConc::__set_current( double current )
{
    set < double > (id_(), "current", current);
}
double CaConc::__get_increase() const
{
    double increase;
    get < double > (id_(), "increase",increase);
    return increase;
}
void CaConc::__set_increase( double increase )
{
    set < double > (id_(), "increase", increase);
}
double CaConc::__get_decrease() const
{
    double decrease;
    get < double > (id_(), "decrease",decrease);
    return decrease;
}
void CaConc::__set_decrease( double decrease )
{
    set < double > (id_(), "decrease", decrease);
}
double CaConc::__get_basalMsg() const
{
    double basalMsg;
    get < double > (id_(), "basalMsg",basalMsg);
    return basalMsg;
}
void CaConc::__set_basalMsg( double basalMsg )
{
    set < double > (id_(), "basalMsg", basalMsg);
}
#endif
