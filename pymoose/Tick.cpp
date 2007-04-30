#ifndef _pymoose_Tick_cpp
#define _pymoose_Tick_cpp
#include "Tick.h"
const std::string Tick::className = "Tick";
Tick::Tick(Id id):PyMooseBase(id){}
Tick::Tick(std::string path):PyMooseBase(className, path){}
Tick::Tick(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
Tick::Tick(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
Tick::~Tick(){}
const std::string& Tick::getType(){ return className; }
double Tick::__get_dt() const
{
    double dt;
    get < double > (Element::element(id_), "dt",dt);
    return dt;
}
void Tick::__set_dt( double dt )
{
    set < double > (Element::element(id_), "dt", dt);
}
int Tick::__get_stage() const
{
    int stage;
    get < int > (Element::element(id_), "stage",stage);
    return stage;
}
void Tick::__set_stage( int stage )
{
    set < int > (Element::element(id_), "stage", stage);
}
int Tick::__get_ordinal() const
{
    int ordinal;
    get < int > (Element::element(id_), "ordinal",ordinal);
    return ordinal;
}
void Tick::__set_ordinal( int ordinal )
{
    set < int > (Element::element(id_), "ordinal", ordinal);
}
double Tick::__get_nextTime() const
{
    double nextTime;
    get < double > (Element::element(id_), "nextTime",nextTime);
    return nextTime;
}
void Tick::__set_nextTime( double nextTime )
{
    set < double > (Element::element(id_), "nextTime", nextTime);
}
std::string Tick::__get_path() const
{
    string path;
    get < string > (Element::element(id_), "path",path);
    return path;
}
void Tick::__set_path( std::string path )
{
    set < string > (Element::element(id_), "path", path);
}
double Tick::__get_updateDtSrc() const
{
    double updateDtSrc;
    get < double > (Element::element(id_), "updateDtSrc",updateDtSrc);
    return updateDtSrc;
}
void Tick::__set_updateDtSrc( double updateDtSrc )
{
    set < double > (Element::element(id_), "updateDtSrc", updateDtSrc);
}
#endif
