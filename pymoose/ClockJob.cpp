#ifndef _pymoose_ClockJob_cpp
#define _pymoose_ClockJob_cpp
#include "ClockJob.h"
const std::string ClockJob::className = "ClockJob";
ClockJob::ClockJob(Id id):PyMooseBase(id){}
ClockJob::ClockJob(std::string path):PyMooseBase(className, path){}
ClockJob::ClockJob(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
ClockJob::ClockJob(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
ClockJob::~ClockJob(){}
const std::string& ClockJob::getType(){ return className; }
double ClockJob::__get_runTime() const
{
    double runTime;
    get < double > (Element::element(id_), "runTime",runTime);
    return runTime;
}
void ClockJob::__set_runTime( double runTime )
{
    set < double > (Element::element(id_), "runTime", runTime);
}
double ClockJob::__get_currentTime() const
{
    double currentTime;
    get < double > (Element::element(id_), "currentTime",currentTime);
    return currentTime;
}
void ClockJob::__set_currentTime( double currentTime )
{
    set < double > (Element::element(id_), "currentTime", currentTime);
}
int ClockJob::__get_nsteps() const
{
    int nsteps;
    get < int > (Element::element(id_), "nsteps",nsteps);
    return nsteps;
}
void ClockJob::__set_nsteps( int nsteps )
{
    set < int > (Element::element(id_), "nsteps", nsteps);
}
int ClockJob::__get_currentStep() const
{
    int currentStep;
    get < int > (Element::element(id_), "currentStep",currentStep);
    return currentStep;
}
void ClockJob::__set_currentStep( int currentStep )
{
    set < int > (Element::element(id_), "currentStep", currentStep);
}
double ClockJob::__get_start() const
{
    double start;
    get < double > (Element::element(id_), "start",start);
    return start;
}
void ClockJob::__set_start( double start )
{
    set < double > (Element::element(id_), "start", start);
}
int ClockJob::__get_step() const
{
    int step;
    get < int > (Element::element(id_), "step",step);
    return step;
}
void ClockJob::__set_step( int step )
{
    set < int > (Element::element(id_), "step", step);
}
#endif
