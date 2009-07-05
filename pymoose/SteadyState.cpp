#ifndef _pymoose_SteadyState_cpp
#define _pymoose_SteadyState_cpp
#include "SteadyState.h"
using namespace pymoose;
const std::string SteadyState::className_ = "SteadyState";
SteadyState::SteadyState(Id id):PyMooseBase(id){}
SteadyState::SteadyState(std::string path):PyMooseBase(className_, path){}
SteadyState::SteadyState(std::string name, Id parentId):PyMooseBase(className_, name, parentId){}
SteadyState::SteadyState(std::string name, PyMooseBase& parent):PyMooseBase(className_, name, parent){}
SteadyState::SteadyState(const SteadyState& src, std::string objectName, PyMooseBase& parent):PyMooseBase(src, objectName, parent){}
SteadyState::SteadyState(const SteadyState& src, std::string objectName, Id& parent):PyMooseBase(src, objectName, parent){}
SteadyState::SteadyState(const SteadyState& src, std::string path):PyMooseBase(src, path){}
SteadyState::SteadyState(const Id& src, std::string name, Id& parent):PyMooseBase(src, name, parent){}
SteadyState::~SteadyState(){}
const std::string& SteadyState::getType(){ return className_; }
bool SteadyState::__get_badStoichiometry() const
{
    bool badStoichiometry;
    get < bool > (id_(), "badStoichiometry",badStoichiometry);
    return badStoichiometry;
}
bool SteadyState::__get_isInitialized() const
{
    bool isInitialized;
    get < bool > (id_(), "isInitialized",isInitialized);
    return isInitialized;
}
unsigned int SteadyState::__get_nIter() const
{
    unsigned int nIter;
    get < unsigned int > (id_(), "nIter",nIter);
    return nIter;
}
unsigned int SteadyState::__get_maxIter() const
{
    unsigned int maxIter;
    get < unsigned int > (id_(), "maxIter",maxIter);
    return maxIter;
}
void SteadyState::__set_maxIter( unsigned int maxIter )
{
    set < unsigned int > (id_(), "maxIter", maxIter);
}
double SteadyState::__get_convergenceCriterion() const
{
    double convergenceCriterion;
    get < double > (id_(), "convergenceCriterion",convergenceCriterion);
    return convergenceCriterion;
}
void SteadyState::__set_convergenceCriterion( double convergenceCriterion )
{
    set < double > (id_(), "convergenceCriterion", convergenceCriterion);
}
unsigned int SteadyState::__get_rank() const
{
    unsigned int rank;
    get < unsigned int > (id_(), "rank",rank);
    return rank;
}
#endif
