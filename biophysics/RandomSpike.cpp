/*******************************************************************
 * File:            RandomSpike.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2008-02-04 11:33:37
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

#ifndef _RANDOMSPIKE_CPP
#define _RANDOMSPIKE_CPP
#include "moose.h"
#include <cmath>
#include "RandomSpike.h"

const Cinfo* initRandomSpikeCinfo()
{
    static Finfo* processShared[] =
        {
            new DestFinfo( "process", Ftype1<ProcInfo>::global(),
                           RFCAST(&RandomSpike::processFunc)),
            new DestFinfo( "reinit", Ftype1<ProcInfo>::global(),
                           RFCAST( &RandomSpike::reinitFunc)),
        };
    static Finfo* process = new SharedFinfo( "process", processShared,
                                             sizeof(processShared)/sizeof(Finfo*));
    static Finfo* randomspikeFinfos[] = 
        {
            new ValueFinfo("minAmp", ValueFtype1<double>::global(),
                           GFCAST( &RandomSpike::getMinAmp),
                           RFCAST( &RandomSpike::setMinAmp)),
            new ValueFinfo("maxAmp", ValueFtype1<double>::global(),
                           GFCAST( &RandomSpike::getMaxAmp),
                           RFCAST( &RandomSpike::setMaxAmp)),
            new ValueFinfo("rate", ValueFtype1<double>::global(),
                           GFCAST( &RandomSpike::getRate),
                           RFCAST( &RandomSpike::setRate)),    
            new ValueFinfo("resetValue", ValueFtype1<double>::global(),
                           GFCAST( &RandomSpike::getResetValue),
                           RFCAST( &RandomSpike::setResetValue)),
            new ValueFinfo("state", ValueFtype1<double>::global(),
                           GFCAST( &RandomSpike::getState),
                           RFCAST( &RandomSpike::setState)),
            new ValueFinfo("absRefract", ValueFtype1<double>::global(),
                           GFCAST( &RandomSpike::getAbsRefract),
                           RFCAST( &RandomSpike::setAbsRefract)),
            new ValueFinfo("lastEvent", ValueFtype1<double>::global(),
                           GFCAST( &RandomSpike::getLastEvent),
                           &dummyFunc),
            new ValueFinfo("reset", ValueFtype1<int>::global(),
                           GFCAST( &RandomSpike::getReset),
                           RFCAST( &RandomSpike::setReset)),                                       
      
            //////////////////////////////////////////////////////////////////
            // SharedFinfos
            //////////////////////////////////////////////////////////////////
            process,

            ///////////////////////////////////////////////////////
            // MsgSrc definitions
            ///////////////////////////////////////////////////////
            new SrcFinfo("event", Ftype1<double>::global()),
            
            //////////////////////////////////////////////////////////////////
            // MessageDestinations
            //////////////////////////////////////////////////////////////////
            new DestFinfo("rateDest", Ftype1<double>::global(),
                          RFCAST(&RandomSpike::setRate)),
            new DestFinfo("minAmpDest", Ftype1<double>::global(),
                          RFCAST( &RandomSpike::setMinAmp)),
            new DestFinfo("maxAmpDest", Ftype1<double>::global(),
                          RFCAST( &RandomSpike::setMaxAmp)),
            new DestFinfo("minmaxDest", Ftype2<double, double>::global(),
                          RFCAST( & RandomSpike::setMinMaxAmp)),
        };
    static Cinfo randomSpikeCinfo("RandomSpike",
                               "Subhasis Ray, 2008, NCBS",
                                  "RandomSpike: generates random events",
                               initNeutralCinfo(),
                               randomspikeFinfos,
                               sizeof(randomspikeFinfos)/sizeof(Finfo*),
                               ValueFtype1<RandomSpike>::global());
    return &randomSpikeCinfo;
}

static const Cinfo* randomSpikeCinfo = initRandomSpikeCinfo();

static const Slot eventSlot = initRandomSpikeCinfo()->getSlot( "event");


RandomSpike::RandomSpike()
{
    minAmp_ = 0.0;
    maxAmp_ = 0.0;
    rate_ = 0.0;
    reset_ = 0;
    resetValue_ = 0.0;
    state_ = 0.0;
    absRefract_ = 0.0;
    lastEvent_ = 0.0;
}


void RandomSpike::setMinAmp(const Conn* c, double value)
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT(obj != NULL, "RandomSpike::setMinAmp(const Conn&, double) - target data pointer is NULL.");
    obj->minAmp_ = value;    
}

double RandomSpike::getMinAmp(Eref e)
{
    RandomSpike* obj = static_cast <RandomSpike*> (e.data());
    ASSERT( obj != NULL, "RandomSpike::getMinAmp(Eref ) - target data pointer is NULL." );
    return obj->minAmp_;    
}
void RandomSpike::setMaxAmp(const Conn* c, double value)
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT(obj != NULL, "RandomSpike::setMaxAmp(const Conn&, double) - target data pointer is NULL.");
    obj->maxAmp_ = value;    
}
double RandomSpike::getMaxAmp(Eref e)
{
    RandomSpike* obj = static_cast <RandomSpike*> (e.data());
    ASSERT( obj != NULL, "RandomSpike::getMaxAmp(Eref ) - target data pointer is NULL." );
    return obj->maxAmp_;    
}
void RandomSpike::setRate(const Conn* c, double value)
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT(obj != NULL, "RandomSpike::setRate(const Conn&, double) - target data pointer is NULL.");
    obj->rate_ = value;    
}
double RandomSpike::getRate(Eref e)
{
    RandomSpike* obj = static_cast <RandomSpike*> (e.data());
    ASSERT( obj != NULL, "RandomSpike::getRate(Eref ) - target data pointer is NULL." );
    return obj->rate_;    
}
void RandomSpike::setResetValue(const Conn* c, double value)
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT(obj != NULL, "RandomSpike::setResetValue(const Conn&, double) - target data pointer is NULL.");
    obj->resetValue_ = value;    
}
double RandomSpike::getResetValue(Eref e)
{
    RandomSpike* obj = static_cast <RandomSpike*> (e.data());
    ASSERT( obj != NULL, "RandomSpike::getResetValue(Eref ) - target data pointer is NULL." );
    return obj->resetValue_;    
}
void RandomSpike::setState(const Conn* c, double value)
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT(obj != NULL, "RandomSpike::setState(const Conn&, double) - target data pointer is NULL.");
    obj->state_ = value;    
}
double RandomSpike::getState(Eref e)
{
    RandomSpike* obj = static_cast <RandomSpike*> (e.data());
    ASSERT( obj != NULL, "RandomSpike::getState(Eref ) - target data pointer is NULL." );
    return obj->state_;    
}
void RandomSpike::setAbsRefract(const Conn* c, double value)
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT(obj != NULL, "RandomSpike::setAbsRefract(const Conn&, double) - target data pointer is NULL.");
    obj->absRefract_ = value;    
}
double RandomSpike::getAbsRefract(Eref e)
{
    RandomSpike* obj = static_cast <RandomSpike*> (e.data());
    ASSERT( obj != NULL, "RandomSpike::getAbsRefract(Eref ) - target data pointer is NULL." );
    return obj->absRefract_;    
}
void RandomSpike::setLastEvent(const Conn* c, double value)
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT(obj != NULL, "RandomSpike::setLastEvent(const Conn&, double) - target data pointer is NULL.");
    obj->lastEvent_ = value;    
}
double RandomSpike::getLastEvent(Eref e)
{
    RandomSpike* obj = static_cast <RandomSpike*> (e.data());
    ASSERT( obj != NULL, "RandomSpike::getLastEvent(Eref ) - target data pointer is NULL." );
    return obj->lastEvent_;    
}
void RandomSpike::setReset(const Conn* c, int value)
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT(obj != NULL, "RandomSpike::setReset(const Conn&, double) - target data pointer is NULL.");
    obj->reset_ = value;    
}
int RandomSpike::getReset(Eref e)
{
    RandomSpike* obj = static_cast <RandomSpike*> (e.data());
    ASSERT( obj != NULL, "RandomSpike::getReset(Eref ) - target data pointer is NULL." );
    return obj->reset_;    
}

void RandomSpike::setMinMaxAmp(const Conn* c, double min, double max)
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT(obj != NULL, "RandomSpike::setMinMaxAmp(const Conn&, double, double) - target data pointer is NULL.");
    obj->minAmp_ = min;
    obj->maxAmp_ = max;    
}

void RandomSpike::processFunc( const Conn* c, ProcInfo p )
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT( obj != NULL, "RandomSpikeprocessFunc(const Conn&, ProcInfo) - target data pointer is NULL.");
    obj->innerProcessFunc(c, p);
}


void RandomSpike::reinitFunc( const Conn* c, ProcInfo p )
{
    RandomSpike* obj = static_cast<RandomSpike*> (c->data());
    ASSERT( obj != NULL, "RandomSpike::reinitFunc(const Conn&, ProcInfo) - target data pointer is NULL.");
    obj->state_ = obj->resetValue_;
    obj->lastEvent_ = - ( obj->absRefract_);    
}

void RandomSpike::innerProcessFunc(const Conn* c, ProcInfo p)
{
    double t = p->currTime_;
    
    if ( reset_ )
    {
        state_ = resetValue_;
    }
    if ( absRefract_ > t - lastEvent_ )
    {
        return;
    }

    double prob = rate_*p->dt_;
    if ( prob >= 1 || prob > mtrand())
    {
        lastEvent_ = t;
        if (!isEqual(minAmp_,maxAmp_))
        {
            state_ = mtrand()*( maxAmp_ - minAmp_ ) + minAmp_;
        }
        else
        {
            state_ = minAmp_;
        }        
    }
    send1 <double> ( c->target(), eventSlot, state_);    
}

#endif
