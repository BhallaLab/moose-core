#ifndef _pymoose_PulseGen_cpp
#define _pymoose_PulseGen_cpp
#include "PulseGen.h"
using namespace pymoose;
const std::string PulseGen::className_ = "PulseGen";
PulseGen::PulseGen(Id id):Neutral(id){}
PulseGen::PulseGen(std::string path):Neutral(className_, path){}
PulseGen::PulseGen(std::string name, Id parentId):Neutral(className_, name, parentId){}
PulseGen::PulseGen(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
PulseGen::PulseGen(const PulseGen& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
PulseGen::PulseGen(const PulseGen& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
PulseGen::PulseGen(const PulseGen& src, std::string path):Neutral(src, path){}
PulseGen::PulseGen(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
PulseGen::PulseGen(const Id& src, std::string path):Neutral(src, path){}
PulseGen::~PulseGen(){}
const std::string& PulseGen::getType(){ return className_; }
double PulseGen::__get_firstLevel() const
{
    double firstLevel;
    get < double > (id_(), "firstLevel",firstLevel);
    return firstLevel;
}
void PulseGen::__set_firstLevel( double firstLevel )
{
    set < double > (id_(), "firstLevel", firstLevel);
}
double PulseGen::__get_firstWidth() const
{
    double firstWidth;
    get < double > (id_(), "firstWidth",firstWidth);
    return firstWidth;
}
void PulseGen::__set_firstWidth( double firstWidth )
{
    set < double > (id_(), "firstWidth", firstWidth);
}
double PulseGen::__get_firstDelay() const
{
    double firstDelay;
    get < double > (id_(), "firstDelay",firstDelay);
    return firstDelay;
}
void PulseGen::__set_firstDelay( double firstDelay )
{
    set < double > (id_(), "firstDelay", firstDelay);
}
double PulseGen::__get_secondLevel() const
{
    double secondLevel;
    get < double > (id_(), "secondLevel",secondLevel);
    return secondLevel;
}
void PulseGen::__set_secondLevel( double secondLevel )
{
    set < double > (id_(), "secondLevel", secondLevel);
}
double PulseGen::__get_secondWidth() const
{
    double secondWidth;
    get < double > (id_(), "secondWidth",secondWidth);
    return secondWidth;
}
void PulseGen::__set_secondWidth( double secondWidth )
{
    set < double > (id_(), "secondWidth", secondWidth);
}
double PulseGen::__get_secondDelay() const
{
    double secondDelay;
    get < double > (id_(), "secondDelay",secondDelay);
    return secondDelay;
}
void PulseGen::__set_secondDelay( double secondDelay )
{
    set < double > (id_(), "secondDelay", secondDelay);
}
double PulseGen::__get_baseLevel() const
{
    double baseLevel;
    get < double > (id_(), "baseLevel",baseLevel);
    return baseLevel;
}
void PulseGen::__set_baseLevel( double baseLevel )
{
    set < double > (id_(), "baseLevel", baseLevel);
}
double PulseGen::__get_output() const
{
    double output;
    get < double > (id_(), "output",output);
    return output;
}
double PulseGen::__get_trigTime() const
{
    double trigTime;
    get < double > (id_(), "trigTime",trigTime);
    return trigTime;
}
void PulseGen::__set_trigTime(double trigTime)
{
    set <double>(id_(), "trigTime", trigTime);
}

int PulseGen::__get_trigMode() const
{
    int trigMode;
    get < int > (id_(), "trigMode",trigMode);
    return trigMode;
}
void PulseGen::__set_trigMode( int trigMode )
{
    set < int > (id_(), "trigMode", trigMode);
}
int PulseGen::__get_prevInput() const
{
    int prevInput;
    get < int > (id_(), "prevInput",prevInput);
    return prevInput;
}
int PulseGen::getCount() const
{
    int count;
    get < int > (id_(), "count",count);
    return count;
}
void PulseGen::setCount( int count )
{
    set < int > (id_(), "count", count);
}

double PulseGen::getDelay(int index)
{
    double delay;
    lookupGet <double, int>(id_(), "delay", delay, index);
    return delay;
}

double PulseGen::getWidth(int index)
{
    double width;
    lookupGet <double, int>(id_(), "width", width, index);
    return width;
}

double PulseGen::getLevel(int index)
{
    double level;
    lookupGet <double, int>(id_(), "level", level, index);
    return level;
}

void PulseGen::setDelay(int index, double delay)
{
    lookupSet< double, int>(id_(), "delay", delay, index);
}
void PulseGen::setWidth(int index, double width)
{
    lookupSet< double, int>(id_(), "width", width, index);
}
void PulseGen::setLevel(int index, double level)
{
    lookupSet< double, int>(id_(), "level", level, index);
}

#endif
