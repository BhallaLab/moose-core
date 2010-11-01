/*******************************************************************
 * File:            BinSynchan.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          subhasis at ncbs dot res dot in
 * Created:         2007-12-03 15:14:58
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _pymoose_BinSynchan_cpp
#define _pymoose_BinSynchan_cpp
#include "BinSynchan.h"
using namespace pymoose;
const std::string BinSynchan::className_ = "BinSynchan";
/*
  Be very afraid to touch these constructors. Make sure you
  understand what it means to use a pointer to a member function in
  C++. All the circus here is only to give a clean array-like interface
  to weight, delay, poolSize, releaseP and releaseCount in
  python. Because python's __setitem__ and __getitem__ methods are
  actually used for accessing indexd elements, we have to present these
  fields as instance of classes with these methods. But as inner class
  of BinSynchan, weight, delay, etc. need to have handle on the
  containing BinSynchan object and use its methods.  C++ is bad, very
  very bad in this area.
*/
BinSynchan::BinSynchan(Id id):Neutral(id)
{
    weight = new BinSynchanDoubleArray (this, &BinSynchan::__get_weight, &BinSynchan::__set_weight);
    delay = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,&BinSynchan::__get_delay, &BinSynchan::__set_delay);
    poolSize = new InnerPyMooseIterable<BinSynchan, unsigned int, int > (this,  &BinSynchan::__get_poolSize, &BinSynchan::__set_poolSize);
    releaseP = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,  &BinSynchan::__get_releaseP,  &BinSynchan::__set_releaseP);
    
    releaseCount = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,  &BinSynchan::__get_releaseCount,  &BinSynchan::__set_releaseCount);
}
BinSynchan::BinSynchan(std::string path):Neutral(className_, path)
{
 
    weight = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this, &BinSynchan::__get_weight, &BinSynchan::__set_weight);
    delay = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,&BinSynchan::__get_delay, &BinSynchan::__set_delay);
    poolSize = new InnerPyMooseIterable<BinSynchan, unsigned int, int > (this,  &BinSynchan::__get_poolSize, &BinSynchan::__set_poolSize);
    releaseP = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,  &BinSynchan::__get_releaseP,  &BinSynchan::__set_releaseP);   
    releaseCount = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,  &BinSynchan::__get_releaseCount,  &BinSynchan::__set_releaseCount);
}
BinSynchan::BinSynchan(std::string name, Id parentId):Neutral(className_, name, parentId)
{    
    weight = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this, &BinSynchan::__get_weight, &BinSynchan::__set_weight);
    delay = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,&BinSynchan::__get_delay, &BinSynchan::__set_delay);
    poolSize = new InnerPyMooseIterable<BinSynchan, unsigned int, int > (this,  &BinSynchan::__get_poolSize, &BinSynchan::__set_poolSize);
    releaseP = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,  &BinSynchan::__get_releaseP,  &BinSynchan::__set_releaseP);
    releaseCount = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,  &BinSynchan::__get_releaseCount,  &BinSynchan::__set_releaseCount);
}
BinSynchan::BinSynchan(std::string name, PyMooseBase& parent):Neutral(className_, name, parent)
{    
     weight = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this, &BinSynchan::__get_weight, &BinSynchan::__set_weight);
    delay = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,&BinSynchan::__get_delay, &BinSynchan::__set_delay);
    poolSize = new InnerPyMooseIterable<BinSynchan, unsigned int, int > (this,  &BinSynchan::__get_poolSize, &BinSynchan::__set_poolSize);
    releaseP = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,  &BinSynchan::__get_releaseP,  &BinSynchan::__set_releaseP);
    releaseCount = new InnerPyMooseIterable<BinSynchan, unsigned int, double > (this,  &BinSynchan::__get_releaseCount,  &BinSynchan::__set_releaseCount);
}
BinSynchan::BinSynchan(const BinSynchan& src, std::string objectName,  PyMooseBase& parent):Neutral(src, objectName, parent){}

BinSynchan::BinSynchan(const BinSynchan& src, std::string objectName, Id& parent):Neutral(src, objectName, parent){}
BinSynchan::BinSynchan(const BinSynchan& src, std::string path):Neutral(src, path)
{
}
BinSynchan::BinSynchan(const Id& src, std::string path):Neutral(src, path)
{
}

BinSynchan::BinSynchan(const Id& src, string name, Id& parent):Neutral(src, name, parent)
{
}

    

BinSynchan::~BinSynchan()
{
    delete weight;
    delete delay;
    delete poolSize;
    delete releaseP;
    delete releaseCount;    
}
const std::string& BinSynchan::getType(){ return className_; }
double BinSynchan::__get_Gbar() const
{
    double Gbar;
    get < double > (id_(), "Gbar",Gbar);
    return Gbar;
}
void BinSynchan::__set_Gbar( double Gbar )
{
    set < double > (id_(), "Gbar", Gbar);
}
double BinSynchan::__get_Ek() const
{
    double Ek;
    get < double > (id_(), "Ek",Ek);
    return Ek;
}
void BinSynchan::__set_Ek( double Ek )
{
    set < double > (id_(), "Ek", Ek);
}
double BinSynchan::__get_tau1() const
{
    double tau1;
    get < double > (id_(), "tau1",tau1);
    return tau1;
}
void BinSynchan::__set_tau1( double tau1 )
{
    set < double > (id_(), "tau1", tau1);
}
double BinSynchan::__get_tau2() const
{
    double tau2;
    get < double > (id_(), "tau2",tau2);
    return tau2;
}
void BinSynchan::__set_tau2( double tau2 )
{
    set < double > (id_(), "tau2", tau2);
}
bool BinSynchan::__get_normalizeWeights() const
{
    bool normalizeWeights;
    get < bool > (id_(), "normalizeWeights",normalizeWeights);
    return normalizeWeights;
}
void BinSynchan::__set_normalizeWeights( bool normalizeWeights )
{
    set < bool > (id_(), "normalizeWeights", normalizeWeights);
}
double BinSynchan::__get_Gk() const
{
    double Gk;
    get < double > (id_(), "Gk",Gk);
    return Gk;
}
void BinSynchan::__set_Gk( double Gk )
{
    set < double > (id_(), "Gk", Gk);
}
double BinSynchan::__get_Ik() const
{
    double Ik;
    get < double > (id_(), "Ik",Ik);
    return Ik;
}
void BinSynchan::__set_Ik( double Ik )
{
    set < double > (id_(), "Ik", Ik);
}
unsigned int BinSynchan::__get_numSynapses() const
{
    unsigned int numSynapses;
    get < unsigned int > (id_(), "numSynapses",numSynapses);
    return numSynapses;
}
double BinSynchan::__get_weight(unsigned int index) const
{
    double weight;
    lookupGet < double,unsigned int > (id_(), "weight", weight, index);
    return weight;
}
void BinSynchan::__set_weight( unsigned int index, double weight )
{
    lookupSet < double,unsigned int > (id_(), "weight", weight, index);
}
double BinSynchan::__get_delay(unsigned int index) const
{
    double delay;
    lookupGet < double,unsigned int > (id_(), "delay",delay, index);
    return delay;
}
void BinSynchan::__set_delay( unsigned int index, double delay )
{
    lookupSet < double,unsigned int > (id_(), "delay", delay, index);
}
int BinSynchan::__get_poolSize(unsigned int index) const
{
    int poolSize;
    lookupGet < int,unsigned int > (id_(), "poolSize",poolSize, index);
    return poolSize;
}
void BinSynchan::__set_poolSize(  unsigned int index, int poolSize )
{
    lookupSet < int,unsigned int > (id_(), "poolSize", poolSize, index);
}
double BinSynchan::__get_releaseP(unsigned int index) const
{
    double releaseP;
    lookupGet < double,unsigned int > (id_(), "releaseP",releaseP, index);
    return releaseP;
}
void BinSynchan::__set_releaseP( unsigned int index, double releaseP)
{
    lookupSet < double,unsigned int > (id_(), "releaseP", releaseP, index);
}
double BinSynchan::__get_releaseCount(unsigned int index) const
{
    double releaseCount;
    lookupGet < double,unsigned int > (id_(), "releaseCount", releaseCount, index);
    return releaseCount;
}

// double BinSynchan::__get_IkSrc() const
// {
//     double IkSrc;
//     get < double > (id_(), "IkSrc",IkSrc);
//     return IkSrc;
// }
// void BinSynchan::__set_IkSrc( double IkSrc )
// {
//     set < double > (id_(), "IkSrc", IkSrc);
// }
// double,double BinSynchan::__get_origChannel() const
// {
//     double,double origChannel;
//     get < double,double > (id_(), "origChannel",origChannel);
//     return origChannel;
// }
// void BinSynchan::__set_origChannel( double,double origChannel )
// {
//     set < double,double > (id_(), "origChannel", origChannel);
// }
double BinSynchan::__get_synapse() const
{
    double synapse;
    get < double > (id_(), "synapse",synapse);
    return synapse;
}
void BinSynchan::__set_synapse( double synapse )
{
    set < double > (id_(), "synapse", synapse);
}
double BinSynchan::__get_activation() const
{
    double activation;
    get < double > (id_(), "activation",activation);
    return activation;
}
void BinSynchan::__set_activation( double activation )
{
    set < double > (id_(), "activation", activation);
}
double BinSynchan::__get_modulator() const
{
    double modulator;
    get < double > (id_(), "modulator",modulator);
    return modulator;
}
void BinSynchan::__set_modulator( double modulator )
{
    set < double > (id_(), "modulator", modulator);
}
#endif
