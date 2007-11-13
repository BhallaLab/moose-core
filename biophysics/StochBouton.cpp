/*******************************************************************
 * File:            StochBouton.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-12 15:34:01
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

#ifndef _STOCHBOUTON_CPP
#define _STOCHBOUTON_CPP
#include "StochBouton.h"

const Cinfo* initStochBoutonCinfo()
{
    static Finfo* processShared[] = 
        {
            new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                           RFCAST( &StochBouton::processFunc ) ),
            new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                           RFCAST( &StochBouton::reinitFunc ) ),
	};
    static Finfo* channelShared[] =
        {
            new SrcFinfo( "channel", Ftype2< double, double >::global()),
            new DestFinfo( "Vm", Ftype1 <double>::global(),
                           RFCAST( &StochBouton::channelFunc )),
        };

    static Finfo* process = new SharedFinfo( "process",
                                             processShared, 
                                             sizeof( processShared ) / sizeof( Finfo* ) );
    
    static Finfo* stochBoutonFinfos[] = 
        {
            /**
               Probability of release of individual vesicles.
               Statistically simpler and more efficient model could be
               just generating random numbers to reflect the number of
               vesicles released at a particular presynaptic voltage.
               But if we want realistic model the release of multiple
               vesicles over a short interval should be taken into
               account. Each of the release events building up the
               voltage in postsynaptic terminal.

               vesicleReleaseP and releaseCountSrc should be mutually
               exclusive - as we are summing up the behaviour of
               individual vesicles in terms of the distribution
               obtained from releaseCountSrc.
            */
            new ValueFinfo("vesicleP", Ftype1< double >::global(),
                           GFCAST(&StochBouton::getVesicleP),
                           RFCAST(&StochBouton::setVesicleP)),
            /**
               Number of ready to release vesicles - what about
               refilling?
            */
            new ValueFinfo("poolSize", Ftype1< double >::global(),
                           GFCAST(&StochBouton::getPoolSize),
                           RFCAST(&StochBouton::setPoolSize)),
            /// Pool size may be passed as parameter to distribution generator
            new SrcFinfo("poolSizeSrc", Ftype1< double >::global()),
            /**
               This should come from a random number generator which
               generates random numbers with the distribution
               representing the number of vesicles released for the
               given voltage over a time interval in which the release
               will affect the post synaptic voltage.
            */
            new DestFinfo("releaseCountDest", Ftype1 < double >::global(),
                          RFCAST(&StochBouton::setReleaseCount)),
            /**
               This is considered by the post synaptic terminal to
               build up the EPSP. Should we also consider some kind of
               attenuation due to most neurotransmitter molecules
               missing the target?
            */
            new SrcFinfo("releaseCount", Ftype1 < double >::global()),

            /**
               This is for inserting new vesicles into the ready to release pool
            */
            new DestFinfo("incrementPool", Ftype1 < double >::global(),
                          RFCAST(&StochBouton::incrementPool)),
            process,
            new SharedFinfo("channel", channelShared,
                            sizeof(channelShared) / sizeof(Finfo*)),
        };
    /**
       This has been put arbitrarily imitating SynChan - needs more thought
    */
    static SchedInfo schedInfo[] = 
        {
            {
                process, 0, 1
            }
        };
    
    
    static Cinfo stochBoutonCinfo(
        "StochBouton",
        "Subhasis Ray, 2007, NCBS",
        "StochBouton: Synaptic channel object implementing synaptic bouton.\nIncorporates the idea of probabilistic release of vesicle contents.\n",
        initNeutralCinfo(),
        stochBoutonFinfos,
        sizeof(stochBoutonFinfos)/sizeof(Finfo *),
        ValueFtype1 <StochBouton>::global(),
        schedInfo, 1);
    return &stochBoutonCinfo;    
}

static const Cinfo* stochBoutonCinfo = initStochBoutonCinfo();

static const unsigned int channelSlot = stochBoutonCinfo->getSlotIndex("channel.channel");
static const unsigned int releaseCountSlot = stochBoutonCinfo->getSlotIndex("releaseCount");
static const unsigned int poolSizeSlot = stochBoutonCinfo->getSlotIndex("poolSizeSrc");


StochBouton::StochBouton()
{    
}


void StochBouton::channelFunc(const Conn& c, double Vm)
{
    static_cast<StochBouton*>(c.data())->Vm_ = Vm;    
}

double StochBouton::getPoolSize(const Element* e)
{
    return static_cast<StochBouton*>(e->data())->poolSize_;
}

void StochBouton::setPoolSize( const Conn& c, double poolSize)
{
    if ( poolSize < 0 )
    {
        cerr << "ERROR: StochBouton::setPoolSize - trying to set poolSize < 0. " << endl;
        return;        
    }        
    static_cast < StochBouton* >(c.data())->poolSize_ = poolSize;    
}

double StochBouton::getVesicleP(const Element* e)
{
    return static_cast < StochBouton * > (e->data())->vesicleP_;
}

void StochBouton::setVesicleP(const Conn & c, double p)
{
    if ( p < 0 )
    {
        cerr << "ERROR: StochBouton::setVesicleP - trying to set vesicle release probability < 0." << endl;
    }
    static_cast<StochBouton*> (c.data())->vesicleP_ = p;
}

void StochBouton::setReleaseCount(const Conn& c, double count)
{
    static_cast<StochBouton*> (c.data())->releaseCount_ = count;
}

void StochBouton::incrementPool(const Conn& c, double count)
{
    StochBouton* b = static_cast <StochBouton*>(c.data());
    if ( b )
    {        
        b->poolSize_ += count;
        if ( b->poolSize_ < 0 )
        {
            b->poolSize_ = 0;
        }        
    }
    else
    {
        cerr << "ERROR: StochBouton::incrementPool - the StochBouton object is NULL." << endl;
    }    
}

void StochBouton::processFunc(const Conn& c, ProcInfo p)
{
    Element* e = c.targetElement();
    
    if ( e )
    {
        StochBouton* b = static_cast < StochBouton*>(e->data());
        if ( b  )
        {
            // send release count to all destinations
            send1( e, releaseCountSlot, b->releaseCount_);
            // update available vesicle pool size
            if ( b->poolSize_ > b->releaseCount_ )
            {
                b->poolSize_ -= b->releaseCount_;
            }
            else 
            {
                b->poolSize_ = 0;
            }
            // send updated value of pool size to all destinations
            send1( e, poolSizeSlot, b->poolSize_);            
        }
        else 
        {
            cerr << "ERROR: StochBouton::processFunc - target StochBouton object is NULL for element " << e->name() << "." << endl;
        }    
    }
    else 
    {
        cerr << "ERROR: StochBouton::processFunc - target element is NULL." << endl;
    }
}
void StochBouton::reinitFunc(const Conn& c, ProcInfo p)
{
}

#endif
