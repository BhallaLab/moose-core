#ifndef _pymoose_SynChan_cpp
#define _pymoose_SynChan_cpp
#include "SynChan.h"
const std::string SynChan::className = "SynChan";
SynChan::SynChan(Id id):PyMooseBase(id){}
SynChan::SynChan(std::string path):PyMooseBase(className, path){}
SynChan::SynChan(std::string name, Id parentId):PyMooseBase(className, name, parentId){}
SynChan::SynChan(std::string name, PyMooseBase* parent):PyMooseBase(className, name, parent){}
SynChan::~SynChan(){}
const std::string& SynChan::getType(){ return className; }
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
void SynChan::__set_Ik( double Ik )
{
    set < double > (id_(), "Ik", Ik);
}
unsigned int SynChan::__get_numSynapses() const
{
    unsigned int numSynapses;
    get < unsigned int > (id_(), "numSynapses",numSynapses);
    return numSynapses;
}
void SynChan::__set_numSynapses( unsigned int numSynapses )
{
    set < unsigned int > (id_(), "numSynapses", numSynapses);
}
double SynChan::__get_weight() const
{
    double weight;
    get < double > (id_(), "weight",weight);
    return weight;
}
void SynChan::__set_weight( double weight )
{
    set < double > (id_(), "weight", weight);
}
double SynChan::__get_delay() const
{
    double delay;
    get < double > (id_(), "delay",delay);
    return delay;
}
void SynChan::__set_delay( double delay )
{
    set < double > (id_(), "delay", delay);
}
double SynChan::__get_IkSrc() const
{
    double IkSrc;
    get < double > (id_(), "IkSrc",IkSrc);
    return IkSrc;
}
void SynChan::__set_IkSrc( double IkSrc )
{
    set < double > (id_(), "IkSrc", IkSrc);
}
double SynChan::__get_synapse() const
{
    double synapse;
    get < double > (id_(), "synapse",synapse);
    return synapse;
}
void SynChan::__set_synapse( double synapse )
{
    set < double > (id_(), "synapse", synapse);
}
double SynChan::__get_activation() const
{
    double activation;
    get < double > (id_(), "activation",activation);
    return activation;
}
void SynChan::__set_activation( double activation )
{
    set < double > (id_(), "activation", activation);
}
double SynChan::__get_modulator() const
{
    double modulator;
    get < double > (id_(), "modulator",modulator);
    return modulator;
}
void SynChan::__set_modulator( double modulator )
{
    set < double > (id_(), "modulator", modulator);
}
#endif
