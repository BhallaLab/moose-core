// KineticChan.cpp --- 
// 
// Filename: KineticChan.cpp
// Description: 
// Author: Aditya Gilra
// Maintainer: 
// Created: Mon Apr 26 
// Version: 
// Last-Updated: Mon Apr 26
//           By: Aditya Gilra
//     Update #: -
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
//  This implements a computationally efficient model of a synapse
//  using kinetic model of receptor binding
//  based on Dextexhe et al. 1994, Neural Computation.
// 

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <math.h>
#include <queue>
#include "basecode/header.h"
#include "basecode/moose.h"
#include "utility/NumUtil.h"

#include "SynInfo.h"
#include "KineticChan.h"

const Cinfo* initKineticChanCinfo()
{

    static Finfo* KineticChanFinfos[] =
            {
                new ValueFinfo("TrPulseTime", ValueFtype1< double >::global(),
                               GFCAST(&KineticChan::getTrPulseTime),
                               RFCAST(&KineticChan::setTrPulseTime),
                               "NeuroTransmitter Pulse Time"),
            
                ///////////////////////////////////////////////////////
                // MsgSrc definitions
                ///////////////////////////////////////////////////////

                ///////////////////////////////////////////////////////
                // MsgDest definitions
                ///////////////////////////////////////////////////////
            };


	static string doc[] =
	{
		"Name", "KineticChan",
		"Author", "Aditya Gilra, 2010, NCBS",
		"Description", "KineticChan: Synaptic channel incorporating kinetic model of receptor binding. "
		        "Handles saturation of synapse. Incorporates weight and delay. Does not "
				"handle activity-dependent modification, see HebbSynChan for that. "
	};

	static Cinfo KineticChanCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initSynChanCinfo(),
		KineticChanFinfos,
		sizeof( KineticChanFinfos )/sizeof(Finfo *),
		ValueFtype1< KineticChan >::global());

	return &KineticChanCinfo;
}

static const Cinfo* KineticChanCinfo = initKineticChanCinfo();

static const Slot channelSlot =
	initKineticChanCinfo()->getSlot( "channel.channel" );
static const Slot origChannelSlot =
	initKineticChanCinfo()->getSlot( "origChannel" );
static const Slot gkSlot =
	initKineticChanCinfo()->getSlot( "GkSrc" );
static const Slot ikSlot =
	initKineticChanCinfo()->getSlot( "IkSrc" );
static const Slot synapseSlot =
	initKineticChanCinfo()->getSlot( "synapse" );

/**
   Default constructor sets all the constant values in the expression
*/

KineticChan::KineticChan(): TrPulseTime_(1.0)
{
}

/**
   set the Neurotransmitter pulse time for the pre-synaptic part of the channel.
*/

void KineticChan::setTrPulseTime(const Conn* conn, double value)
{
    static_cast<KineticChan*> (conn->data())->innerSetTrPulseTime(value);
}

void KineticChan::innerSetTrPulseTime(double value)
{
    TrPulseTime_ = value;
}

double KineticChan::getTrPulseTime(Eref e)
{
    return static_cast<KineticChan*>(e.data())->innerGetTrPulseTime();
}

double KineticChan::innerGetTrPulseTime()
{
    return TrPulseTime_;
}

void SynChan::innerProcessFunc( Eref e, ProcInfo info )
{
	while ( !pendingEvents_.empty() &&
		pendingEvents_.top().delay <= info->currTime_ ) {
		activation_ += pendingEvents_.top().weight / info->dt_;
		pendingEvents_.pop();
	}
	X_ = modulation_ * activation_ * xconst1_ + X_ * xconst2_;
	Y_ = X_ * yconst1_ + Y_ * yconst2_;
	Gk_ = Y_ * norm_;
	Ik_ = ( Ek_ - Vm_ ) * Gk_;
	activation_ = 0.0;
	modulation_ = 1.0;
	send2< double, double >( e, channelSlot, Gk_, Ek_ );
	send2< double, double >( e, origChannelSlot, Gk_, Ek_ );
	send1< double >( e, ikSlot, Ik_ );
	// Usually needed by GHK-type objects
	send1< double >( e, gkSlot, Gk_ );
}

/*
 * Note that this causes issues if we have variable dt.
 */
void SynChan::innerReinitFunc( Eref e, ProcInfo info )
{
	double dt = info->dt_;
	activation_ = 0.0;
	modulation_ = 1.0;
	Gk_ = 0.0;
	Ik_ = 0.0;
	X_ = 0.0;
	Y_ = 0.0;
	xconst1_ = tau1_ * ( 1.0 - exp( -dt / tau1_ ) );
	xconst2_ = exp( -dt / tau1_ );
        if (isEqual(tau2_, 0.0)) {
                yconst1_ = 1.0;
                yconst2_ = 0.0;
                norm_ = 1.0;
        } else {
                yconst1_ = tau2_ * ( 1.0 - exp( -dt / tau2_ ) );
                yconst2_ = exp( -dt / tau2_ );
                if ( tau1_ == tau2_ ) {
                    norm_ = Gbar_ * SynE / tau1_;
                } else {
                    double tpeak = tau1_ * tau2_ * log( tau1_ / tau2_ ) / 
                            ( tau1_ - tau2_ );
                    norm_ = Gbar_ * ( tau1_ - tau2_ ) / 
                            ( tau1_ * tau2_ * ( 
                                    exp( -tpeak / tau1_ ) - exp( -tpeak / tau2_ )
                                                ));
                }
        }
        
	updateNumSynapse( e );
	if ( normalizeWeights_ && synapses_.size() > 0 )
		norm_ /= static_cast< double >( synapses_.size() );
	while ( !pendingEvents_.empty() )
		pendingEvents_.pop();
}
