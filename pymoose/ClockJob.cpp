
#ifndef _pymoose_ClockJob_cpp
#define _pymoose_ClockJob_cpp
#include "ClockJob.h"
using namespace pymoose;
const std::string ClockJob::className_ = "ClockJob";
ClockJob::ClockJob(Id id):Neutral(id){}
ClockJob::ClockJob(std::string path):Neutral(className_, path){}
ClockJob::ClockJob(std::string name, Id parentId):Neutral(className_, name, parentId){}
ClockJob::ClockJob(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
ClockJob::ClockJob(const ClockJob& src, std::string objectName,  PyMooseBase& parent):Neutral(src, objectName, parent){}

ClockJob::ClockJob(const ClockJob& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
ClockJob::ClockJob(const ClockJob& src, std::string path):Neutral(src, path)
{
}
ClockJob::ClockJob(const Id& src, std::string path):Neutral(src, path)
{
}

ClockJob::ClockJob(const Id& src, string name, Id& parent):Neutral(src, name, parent)
{
}
ClockJob::~ClockJob(){}
const std::string& ClockJob::getType(){ return className_; }
double ClockJob::__get_runTime() const
{
    double runTime;
    get < double > (id_(), "runTime",runTime);
    return runTime;
}
void ClockJob::__set_runTime( double runTime )
{
    set < double > (id_(), "runTime", runTime);
}
double ClockJob::__get_currentTime() const
{
    double currentTime;
    get < double > (id_(), "currentTime",currentTime);
    return currentTime;
}

int ClockJob::__get_nsteps() const
{
    int nsteps;
    get < int > (id_(), "nsteps",nsteps);
    return nsteps;
}
void ClockJob::__set_nsteps( int nsteps )
{
    set < int > (id_(), "nsteps", nsteps);
}
int ClockJob::__get_currentStep() const
{
    int currentStep;
    get < int > (id_(), "currentStep",currentStep);
    return currentStep;
}

int ClockJob::__get_autoschedule() const
{
    int autoschedule;
    get<int>(id_(), "autoschedule", autoschedule);
    return autoschedule;
}

void ClockJob::__set_autoschedule(int value)
{
    set<int>(id_(), "autoschedule", value);
}
// void ClockJob::__set_currentStep( int currentStep )
// {
//     set < int > (id_(), "currentStep", currentStep);
// }
// double ClockJob::__get_start() const
// {
//     double start;
//     get < double > (id_(), "start",start);
//     return start;
// }
// void ClockJob::__set_start( double start )
// {
//     set < double > (id_(), "start", start);
// }
// int ClockJob::__get_step() const
// {
//     int step;
//     get < int > (id_(), "step",step);
//     return step;
// }
// void ClockJob::__set_step( int step )
// {
//     set < int > (id_(), "step", step);
// }

// void ClockJob::resched()
// {
//     set (id_(), "resched");    
// }

// void ClockJob::reinit()
// {
//         set (id_(), "reinit");
// }

// void ClockJob::stop()
// {
//         set (id_(), "stop");
// }

vector <double> & ClockJob::getClocks()
{
    return PyMooseBase::getContext()->getClocks();
}

#endif
