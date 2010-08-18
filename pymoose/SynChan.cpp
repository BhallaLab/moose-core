#ifndef _pymoose_SynChan_cpp
#define _pymoose_SynChan_cpp
#include "SynChan.h"
using namespace pymoose;
const std::string SynChan::className_ = "SynChan";
SynChan::SynChan(Id id):Neutral(id){}
SynChan::SynChan(std::string path):Neutral(className_, path){}
SynChan::SynChan(std::string name, Id parentId):Neutral(className_, name, parentId){}
SynChan::SynChan(std::string name, PyMooseBase& parent):Neutral(className_, name, parent){}
SynChan::SynChan(const SynChan& src, std::string objectName, PyMooseBase& parent):Neutral(src, objectName, parent){}
SynChan::SynChan(const SynChan& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
SynChan::SynChan(const SynChan& src, std::string path):Neutral(src, path){}
SynChan::SynChan(const Id& src, std::string name, Id& parent):Neutral(src, name, parent){}
SynChan::SynChan(const Id& src, std::string path):Neutral(src, path){}
SynChan::~SynChan(){}
const std::string& SynChan::getType(){ return className_; }
double SynChan::__get_Gbar() const
{
    double Gbar;
    get < double > (id_(), "Gbar",Gbar);
    return Gbar;
}
void SynChan::__set_Gbar( double Gbar )
{
    set < double > (id_(), "Gbar", Gbar);
}
double SynChan::__get_Ek() const
{
    double Ek;
    get < double > (id_(), "Ek",Ek);
    return Ek;
}
void SynChan::__set_Ek( double Ek )
{
    set < double > (id_(), "Ek", Ek);
}
double SynChan::__get_tau1() const
{
    double tau1;
    get < double > (id_(), "tau1",tau1);
    return tau1;
}
void SynChan::__set_tau1( double tau1 )
{
    set < double > (id_(), "tau1", tau1);
}
double SynChan::__get_tau2() const
{
    double tau2;
    get < double > (id_(), "tau2",tau2);
    return tau2;
}
void SynChan::__set_tau2( double tau2 )
{
    set < double > (id_(), "tau2", tau2);
}
bool SynChan::__get_normalizeWeights() const
{
    bool normalizeWeights;
    get < bool > (id_(), "normalizeWeights",normalizeWeights);
    return normalizeWeights;
}
void SynChan::__set_normalizeWeights( bool normalizeWeights )
{
    set < bool > (id_(), "normalizeWeights", normalizeWeights);
}
double SynChan::__get_Gk() const
{
    double Gk;
    get < double > (id_(), "Gk",Gk);
    return Gk;
}
void SynChan::__set_Gk( double Gk )
{
    set < double > (id_(), "Gk", Gk);
}
double SynChan::__get_Ik() const
{
    double Ik;
    get < double > (id_(), "Ik",Ik);
    return Ik;
}
unsigned int SynChan::__get_numSynapses() const
{
    unsigned int numSynapses;
    get < unsigned int > (id_(), "numSynapses",numSynapses);
    return numSynapses;
}

// vector<SynInfo>& SynChan::__get_synapse() const
// {
    // This is a problem with the design of SynChan:
    // It has two facets: one as a whole object, second as a list of
    // synapses. The vector of synapses is protected in the class
    // definition, yet it tries to give impression that the user can
    // access individual synapses. In python, it becomes
    // difficult(impossible?) to give the user the impression that
    // synapse is a vector contained inside the SynChan object.
    // Hence the awkward getWeight and setWeight functions.
    // I think a better syntax will be:
    // mySynChan.synapse[0].weight = 1.0
    // rather than mySynChan.setWeight(1.0, 0), which is highly confusing.
//     return 0;
// }
double SynChan::getWeight(const unsigned int& index) const
{
    double weight;
    lookupGet < double, unsigned int > (id_(), "weight", weight, index);
    return weight;
}
void SynChan::setWeight( const unsigned int& index ,double weight )
{
    lookupSet < double, unsigned int > (id_(), "weight", weight, index);
}
double SynChan::getDelay(const unsigned int& index) const
{
    double delay;
    lookupGet < double, unsigned int > (id_.eref(), "delay", delay, index);
    return delay;
}
void SynChan::setDelay( const unsigned int& index, double delay)
{
    lookupSet < double, unsigned int  > (id_.eref(), "delay", delay, index);
}

//Manually edited
// These are for allowing Table access to constructors in PyMooseBase
SynChan::SynChan(std::string typeName, std::string objectName, Id parentId): Neutral(typeName, objectName, parentId)
{
}
   
SynChan::SynChan(std::string typeName, std::string path): Neutral(typeName, path)
{
}

SynChan::SynChan(std::string typeName, std::string objectName, PyMooseBase& parent): Neutral(typeName, objectName, parent)
{
}

#endif
