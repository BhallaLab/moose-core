#ifndef _pymoose_StochSynchan_cpp
#define _pymoose_StochSynchan_cpp
#include "StochSynchan.h"
using namespace pymoose;
const std::string StochSynchan::className_ = "StochSynchan";
const std::string& StochSynchan::getType(){ return className_; }
double StochSynchan::__get_Gbar() const
{
    double Gbar;
    get < double > (id_(), "Gbar",Gbar);
    return Gbar;
}
void StochSynchan::__set_Gbar( double Gbar )
{
    set < double > (id_(), "Gbar", Gbar);
}
double StochSynchan::__get_Ek() const
{
    double Ek;
    get < double > (id_(), "Ek",Ek);
    return Ek;
}
void StochSynchan::__set_Ek( double Ek )
{
    set < double > (id_(), "Ek", Ek);
}
double StochSynchan::__get_tau1() const
{
    double tau1;
    get < double > (id_(), "tau1",tau1);
    return tau1;
}
void StochSynchan::__set_tau1( double tau1 )
{
    set < double > (id_(), "tau1", tau1);
}
double StochSynchan::__get_tau2() const
{
    double tau2;
    get < double > (id_(), "tau2",tau2);
    return tau2;
}
void StochSynchan::__set_tau2( double tau2 )
{
    set < double > (id_(), "tau2", tau2);
}
bool StochSynchan::__get_normalizeWeights() const
{
    bool normalizeWeights;
    get < bool > (id_(), "normalizeWeights",normalizeWeights);
    return normalizeWeights;
}
void StochSynchan::__set_normalizeWeights( bool normalizeWeights )
{
    set < bool > (id_(), "normalizeWeights", normalizeWeights);
}
double StochSynchan::__get_Gk() const
{
    double Gk;
    get < double > (id_(), "Gk",Gk);
    return Gk;
}
void StochSynchan::__set_Gk( double Gk )
{
    set < double > (id_(), "Gk", Gk);
}
double StochSynchan::__get_Ik() const
{
    double Ik;
    get < double > (id_(), "Ik",Ik);
    return Ik;
}
void StochSynchan::__set_Ik( double Ik )
{
    set < double > (id_(), "Ik", Ik);
}
unsigned int StochSynchan::__get_numSynapses() const
{
    unsigned int numSynapses;
    get < unsigned int > (id_(), "numSynapses",numSynapses);
    return numSynapses;
}

double StochSynchan::__get_synapse() const
{
    double synapse;
    get < double > (id_(), "synapse",synapse);
    return synapse;
}
void StochSynchan::__set_synapse( double synapse )
{
    set < double > (id_(), "synapse", synapse);
}
double StochSynchan::__get_activation() const
{
    double activation;
    get < double > (id_(), "activation",activation);
    return activation;
}
void StochSynchan::__set_activation( double activation )
{
    set < double > (id_(), "activation", activation);
}
double StochSynchan::__get_modulator() const
{
    double modulator;
    get < double > (id_(), "modulator",modulator);
    return modulator;
}
void StochSynchan::__set_modulator( double modulator )
{
    set < double > (id_(), "modulator", modulator);
}

// The following functions were manually inserted - so think twice before editing
StochSynchan::StochSynchan(Id id):Neutral(id)
{
    weight = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this, &StochSynchan::__get_weight, &StochSynchan::__set_weight);
    delay = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,&StochSynchan::__get_delay, &StochSynchan::__set_delay);
    releaseP = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,  &StochSynchan::__get_releaseP,  &StochSynchan::__set_releaseP);    
    releaseCount = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,  &StochSynchan::__get_releaseCount,  &StochSynchan::__set_releaseCount);
}
StochSynchan::StochSynchan(std::string path):Neutral(className_, path)
{
 
    weight = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this, &StochSynchan::__get_weight, &StochSynchan::__set_weight);
    delay = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,&StochSynchan::__get_delay, &StochSynchan::__set_delay);
    releaseP = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,  &StochSynchan::__get_releaseP,  &StochSynchan::__set_releaseP);   
    releaseCount = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,  &StochSynchan::__get_releaseCount,  &StochSynchan::__set_releaseCount);
}
StochSynchan::StochSynchan(std::string name, Id parentId):Neutral(className_, name, parentId)
{    
    weight = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this, &StochSynchan::__get_weight, &StochSynchan::__set_weight);
    delay = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,&StochSynchan::__get_delay, &StochSynchan::__set_delay);
    releaseP = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,  &StochSynchan::__get_releaseP,  &StochSynchan::__set_releaseP);
    releaseCount = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,  &StochSynchan::__get_releaseCount,  &StochSynchan::__set_releaseCount);
}
StochSynchan::StochSynchan(std::string name, PyMooseBase& parent):Neutral(className_, name, parent)
{    
     weight = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this, &StochSynchan::__get_weight, &StochSynchan::__set_weight);
    delay = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,&StochSynchan::__get_delay, &StochSynchan::__set_delay);
    releaseP = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,  &StochSynchan::__get_releaseP,  &StochSynchan::__set_releaseP);
    releaseCount = new InnerPyMooseIterable<StochSynchan, unsigned int, double > (this,  &StochSynchan::__get_releaseCount,  &StochSynchan::__set_releaseCount);
}
StochSynchan::StochSynchan(const StochSynchan& src, std::string objectName,  PyMooseBase& parent):Neutral(src, objectName, parent){}

StochSynchan::StochSynchan(const StochSynchan& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
StochSynchan::StochSynchan(const StochSynchan& src, std::string path):Neutral(src, path)
{
}

StochSynchan::StochSynchan(const Id& src, string name, Id& parent):Neutral(src, name, parent)
{
}
StochSynchan::StochSynchan(const Id& src, string path):Neutral(src, path)
{
}
StochSynchan::~StochSynchan()
{
    delete weight;
    delete delay;
    delete releaseP;
    delete releaseCount;    
}


double StochSynchan::__get_weight(unsigned int index) const
{
    double weight;
    lookupGet < double,unsigned int > (id_(), "weight", weight, index);
    return weight;
}
void StochSynchan::__set_weight( unsigned int index, double weight )
{
    lookupSet < double,unsigned int > (id_(), "weight", weight, index);
}
double StochSynchan::__get_delay(unsigned int index) const
{
    double delay;
    lookupGet < double,unsigned int > (id_(), "delay",delay, index);
    return delay;
}
void StochSynchan::__set_delay( unsigned int index, double delay )
{
    lookupSet < double,unsigned int > (id_(), "delay", delay, index);
}
double StochSynchan::__get_releaseP(unsigned int index) const
{
    double releaseP;
    lookupGet < double,unsigned int > (id_(), "releaseP",releaseP, index);
    return releaseP;
}
void StochSynchan::__set_releaseP( unsigned int index, double releaseP)
{
    lookupSet < double,unsigned int > (id_(), "releaseP", releaseP, index);
}
double StochSynchan::__get_releaseCount(unsigned int index) const
{
    double releaseCount;
    lookupGet < double,unsigned int > (id_(), "releaseCount", releaseCount, index);
    return releaseCount;
}

#endif
