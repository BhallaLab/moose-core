#ifndef _pymoose_RC_cpp
#define _pymoose_RC_cpp
#include "RC.h"
using namespace pymoose;
const std::string RC::className_ = "RC";
RC::RC(Id id):Neutral(id){}
RC::RC(std::string path):Neutral(className_, path){}
RC::RC(std::string name, Id parentId):Neutral(className_, name, parentId){}
RC::RC(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
RC::RC(const RC& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
RC::RC(const RC& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
RC::RC(const RC& src, std::string path):Neutral(src, path){}
RC::RC(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
RC::RC(const Id& src, std::string path):Neutral(src, path){}
RC::~RC(){}
const std::string& RC::getType(){ return className_; }
double RC::__get_V0() const
{
    double V0;
    get < double > (id_(), "V0",V0);
    return V0;
}
void RC::__set_V0( double V0 )
{
    set < double > (id_(), "V0", V0);
}
double RC::__get_R() const
{
    double R;
    get < double > (id_(), "R",R);
    return R;
}
void RC::__set_R( double R )
{
    set < double > (id_(), "R", R);
}
double RC::__get_C() const
{
    double C;
    get < double > (id_(), "C",C);
    return C;
}
void RC::__set_C( double C )
{
    set < double > (id_(), "C", C);
}
double RC::__get_state() const
{
    double state;
    get < double > (id_(), "state",state);
    return state;
}
double RC::__get_inject() const
{
    double inject;
    get < double > (id_(), "inject",inject);
    return inject;
}
void RC::__set_inject( double inject )
{
    set < double > (id_(), "inject", inject);
}
#endif
