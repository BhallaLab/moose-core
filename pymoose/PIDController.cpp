#ifndef _pymoose_PIDController_cpp
#define _pymoose_PIDController_cpp
#include "PIDController.h"
using namespace pymoose;
const std::string PIDController::className_ = "PIDController";
PIDController::PIDController(Id id):PyMooseBase(id){}
PIDController::PIDController(std::string path):PyMooseBase(className_, path){}
PIDController::PIDController(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
PIDController::PIDController(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
PIDController::PIDController(const PIDController& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
PIDController::PIDController(const PIDController& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
PIDController::PIDController(const PIDController& src, std::string path):PyMooseBase(src, path){}
PIDController::PIDController(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
PIDController::PIDController(const Id& src, std::string path):PyMooseBase(src, path){}
PIDController::~PIDController(){}
const std::string& PIDController::getType(){ return className_; }
double PIDController::__get_gain() const
{
    double gain;
    get < double > (id_(), "gain",gain);
    return gain;
}
void PIDController::__set_gain( double gain )
{
    set < double > (id_(), "gain", gain);
}
double PIDController::__get_saturation() const
{
    double saturation;
    get < double > (id_(), "saturation",saturation);
    return saturation;
}
void PIDController::__set_saturation( double saturation )
{
    set < double > (id_(), "saturation", saturation);
}
double PIDController::__get_command() const
{
    double command;
    get < double > (id_(), "command",command);
    return command;
}
void PIDController::__set_command( double command )
{
    set < double > (id_(), "command", command);
}
double PIDController::__get_sensed() const
{
    double sensed;
    get < double > (id_(), "sensed",sensed);
    return sensed;
}
double PIDController::__get_tauI() const
{
    double tauI;
    get < double > (id_(), "tauI",tauI);
    return tauI;
}
void PIDController::__set_tauI( double tauI )
{
    set < double > (id_(), "tauI", tauI);
}
double PIDController::__get_tauD() const
{
    double tauD;
    get < double > (id_(), "tauD",tauD);
    return tauD;
}
void PIDController::__set_tauD( double tauD )
{
    set < double > (id_(), "tauD", tauD);
}
double PIDController::__get_output() const
{
    double output;
    get < double > (id_(), "output",output);
    return output;
}
#endif
