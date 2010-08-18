#ifndef _pymoose_Tick_cpp
#define _pymoose_Tick_cpp
#include "Tick.h"
using namespace pymoose;

const std::string Tick::className_ = "Tick";
Tick::Tick(Id id):Neutral(id){}
Tick::Tick(std::string path):Neutral(className_, path){}
Tick::Tick(std::string name, Id parentId):Neutral(className_, name, parentId){}
Tick::Tick(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
Tick::Tick(const Tick& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
Tick::Tick(const Tick& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
Tick::Tick(const Tick& src, std::string path):Neutral(src, path){}
Tick::Tick(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
Tick::Tick(const Id& src, std::string path):Neutral(src, path){}
Tick::~Tick(){}
const std::string& Tick::getType(){ return className_; }
double Tick::__get_dt() const
{
    double dt;
    get < double > (id_(), "dt",dt);
    return dt;
}
void Tick::__set_dt( double dt )
{
    set < double > (id_(), "dt", dt);
}
int Tick::__get_stage() const
{
    int stage;
    get < int > (id_(), "stage",stage);
    return stage;
}
void Tick::__set_stage( int stage )
{
    set < int > (id_(), "stage", stage);
}
int Tick::__get_ordinal() const
{
    int ordinal;
    get < int > (id_(), "ordinal",ordinal);
    return ordinal;
}
void Tick::__set_ordinal( int ordinal )
{
    set < int > (id_(), "ordinal", ordinal);
}
double Tick::__get_nextTime() const
{
    double nextTime;
    get < double > (id_(), "nextTime",nextTime);
    return nextTime;
}
void Tick::__set_nextTime( double nextTime )
{
    set < double > (id_(), "nextTime", nextTime);
}
std::string Tick::__get_path() const
{
    string path;
    get < string > (id_(), "path",path);
    return path;
}
void Tick::__set_path( std::string path )
{
    set < string > (id_(), "path", path);
}
double Tick::__get_updateDtSrc() const
{
    double updateDtSrc;
    get < double > (id_(), "updateDtSrc",updateDtSrc);
    return updateDtSrc;
}
void Tick::__set_updateDtSrc( double updateDtSrc )
{
    set < double > (id_(), "updateDtSrc", updateDtSrc);
}
#endif
