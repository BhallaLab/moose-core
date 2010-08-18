#ifndef _pymoose_TimeTable_cpp
#define _pymoose_TimeTable_cpp
#include "TimeTable.h"
using namespace pymoose;
const std::string TimeTable::className_ = "TimeTable";
TimeTable::TimeTable(Id id):Neutral(id){}
TimeTable::TimeTable(std::string path):Neutral(className_, path){}
TimeTable::TimeTable(std::string name, Id parentId):Neutral(className_, name, parentId){}
TimeTable::TimeTable(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
TimeTable::TimeTable(const TimeTable& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
TimeTable::TimeTable(const TimeTable& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
TimeTable::TimeTable(const TimeTable& src, std::string path):Neutral(src, path){}
TimeTable::TimeTable(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
TimeTable::TimeTable(const Id& src, std::string path):Neutral(src, path){}
TimeTable::~TimeTable(){}
const std::string& TimeTable::getType(){ return className_; }
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
void TimeTable::__set_tableVector( const vector < double >& tableVector )
{
    set < vector < double > > (id_(), "tableVector", tableVector);
}
unsigned int TimeTable::__get_tableSize() const
{
    unsigned int tableSize;
    get < unsigned int > (id_(), "tableSize",tableSize);
    return tableSize;
}

double TimeTable::__getitem__(const unsigned int index) const
{
    double value;
    lookupGet <double, unsigned int>(id_.eref(), "table", value, index);
    return value;
}

void TimeTable::__setitem__(const unsigned int index, double value)
{
    cout << "Warning: TimeTable currently only supports loading table entries from file. This assignement will have no effect. Try setting \"filename\" attribute." << endl;
    lookupSet<double, unsigned int>(id_.eref(), "table", value, index); // this is ineffective
}

double TimeTable::__get_state()
{
    double value;
    get <double> (id_.eref(), "state", value);
    return value;
}

int TimeTable::__get_method()
{
    int value;
    get <int> (id_.eref(), "method", value);
    return value;
}

void TimeTable::__set_method(const int method)
{
    set<int> (id_.eref(), "method", method);
}

const std::string TimeTable::__get_filename()
{
    std::string value;
    get<std::string> (id_.eref(), "filename", value);
    return value;
}

void TimeTable::__set_filename(const std::string& filename)
{
    set<std::string>(id_.eref(), "filename", filename);
}
#endif
