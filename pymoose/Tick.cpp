#ifndef _pymoose_Tick_cpp
#define _pymoose_Tick_cpp
#include "Tick.h"
const std::string ClockTick::className = "ClockTick";
ClockTick::ClockTick(Id id):PyMooseBase(id){}
ClockTick::ClockTick(std::string path):PyMooseBase(className, path){}
ClockTick::ClockTick(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
ClockTick::ClockTick(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
ClockTick::~ClockTick(){}
const std::string& ClockTick::getType(){ return className; }
double ClockTick::__get_dt() const
{
    double dt;
    get < double > (id_(), "dt",dt);
    return dt;
}
void ClockTick::__set_dt( double dt )
{
    set < double > (id_(), "dt", dt);
}
int ClockTick::__get_stage() const
{
    int stage;
    get < int > (id_(), "stage",stage);
    return stage;
}
void ClockTick::__set_stage( int stage )
{
    set < int > (id_(), "stage", stage);
}
int ClockTick::__get_ordinal() const
{
    int ordinal;
    get < int > (id_(), "ordinal",ordinal);
    return ordinal;
}
void ClockTick::__set_ordinal( int ordinal )
{
    set < int > (id_(), "ordinal", ordinal);
}
double ClockTick::__get_nextTime() const
{
    double nextTime;
    get < double > (id_(), "nextTime",nextTime);
    return nextTime;
}
void ClockTick::__set_nextTime( double nextTime )
{
    set < double > (id_(), "nextTime", nextTime);
}
std::string ClockTick::__get_path() const
{
    string path;
    get < string > (id_(), "path",path);
    return path;
}
void ClockTick::__set_path( std::string path )
{
    set < string > (id_(), "path", path);
}
double ClockTick::__get_updateDtSrc() const
{
    double updateDtSrc;
    get < double > (id_(), "updateDtSrc",updateDtSrc);
    return updateDtSrc;
}
void ClockTick::__set_updateDtSrc( double updateDtSrc )
{
    set < double > (id_(), "updateDtSrc", updateDtSrc);
}
#endif
