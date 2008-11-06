#ifndef _pymoose_TimeTable_cpp
#define _pymoose_TimeTable_cpp
#include "TimeTable.h"
using namespace pymoose;
const std::string TimeTable::className = "TimeTable";
TimeTable::TimeTable(Id id):PyMooseBase(id){}
TimeTable::TimeTable(std::string path):PyMooseBase(className, path){}
TimeTable::TimeTable(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
TimeTable::TimeTable(std::string name, PyMooseBase& parent):PyMooseBase(className, name, parent){}
TimeTable::TimeTable(const TimeTable& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
TimeTable::TimeTable(const TimeTable& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
TimeTable::TimeTable(const TimeTable& src, std::string path):PyMooseBase(src, path){}
TimeTable::TimeTable(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
TimeTable::~TimeTable(){}
const std::string& TimeTable::getType(){ return className; }
double TimeTable::__get_maxTime() const
{
    double maxTime;
    get < double > (id_(), "maxTime",maxTime);
    return maxTime;
}
void TimeTable::__set_maxTime( double maxTime )
{
    set < double > (id_(), "maxTime", maxTime);
}
vector < double > TimeTable::__get_tableVector() const
{
    vector < double > tableVector;
    get < vector < double > > (id_(), "tableVector",tableVector);
    return tableVector;
}
void TimeTable::__set_tableVector( vector < double > tableVector )
{
    set < vector < double > > (id_(), "tableVector", tableVector);
}
unsigned int TimeTable::__get_tableSize() const
{
    unsigned int tableSize;
    get < unsigned int > (id_(), "tableSize",tableSize);
    return tableSize;
}
#endif
