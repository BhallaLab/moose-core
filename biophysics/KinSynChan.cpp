// KinSynChan.cpp --- 
// 
// Filename: KinSynChan.cpp
// Description: 
// Author: Aditya Gilra
// Maintainer: 
// Created: Mon Apr 26 
// Version: 
// Last-Updated: Wed Apr 28 15:06:14 2010 (+0530)
//           By: Subhasis Ray
//     Update #: 189
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
#include "KinSynChan.h"

const Cinfo* initKinSynChanCinfo()
{

    static Finfo* KinSynChanFinfos[] =
        {
            new ValueFinfo("rInf", ValueFtype1< double >::global(),
                           GFCAST(&KinSynChan::getRinf),
                           RFCAST(&KinSynChan::setRinf),
                           "The fraction of bound receptors at steady state."),
                           
            new ValueFinfo( "tau1", ValueFtype1< double >::global(),
             	GFCAST( &SynChan::getTau1 ), 
                             RFCAST( &SynChan::setTau1 ),
                             "Decay time constant for the synaptic conductance. This corresponds to 1/beta in [Destexhe, Mainen and Sejnowski, 1994]. You can set this tau1 or rInf, one deciding the value of the other."
             ),
             
		    // new ValueFinfo( "tau2", ValueFtype1< double >::global(),
		    // 	GFCAST( &KinSynChan::getTau2 ), 
                    //         RFCAST( &KinSynChan::setTau2 ),
                    //         "Rise time constant for the synaptic conductance. It is calculated using rInf and tau1 [See: Destexhe, Mainen and Sejnowski, 1994. It corresponds to tau_r in that paper]."
		    // ),
                    
		    // new ValueFinfo( "tauR", ValueFtype1< double >::global(),
			//    GFCAST( &SynChan::getTau2 ), 
            //                RFCAST( &SynChan::setTau2 ),
            //                "Rise time constant for the synaptic conductance. It is calculated using rInf and tau1 [See: Destexhe, Mainen and Sejnowski, 1994. It corresponds to tau_r in that paper.]"
		    // ),

		    new ValueFinfo( "pulseWidth", ValueFtype1< double >::global(),
			    GFCAST( &KinSynChan::getPulseWidth ), 
                            RFCAST( &KinSynChan::setPulseWidth ),
                            "The duration of the neuroTransmitter pulse. [See: Destexhe, Mainen and Sejnowski, 1994. It corresponds to t1 in that paper.]"
		    ),
                
                ///////////////////////////////////////////////////////
                // MsgSrc definitions
                ///////////////////////////////////////////////////////

                ///////////////////////////////////////////////////////
                // MsgDest definitions
                ///////////////////////////////////////////////////////
            };


	static string doc[] =
	{
		"Name", "KinSynChan",
		"Author", "Aditya Gilra and Subhasis Ray, 2010, NCBS",
		"Description", "KinSynChan: Synaptic channel incorporating kinetic model of receptor binding. "
		        "From Destexhe, Mainen and Sejnowski, 1994"
		        "Handles saturation of synapse. Incorporates weight and delay. Does not "
				"handle activity-dependent modification, see HebbSynChan for that. "
	};

	static Cinfo KinSynChanCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initSynChanCinfo(),
		KinSynChanFinfos,
		sizeof( KinSynChanFinfos )/sizeof(Finfo *),
		ValueFtype1< KinSynChan >::global());

	return &KinSynChanCinfo;
}

static const Cinfo* KinSynChanCinfo = initKinSynChanCinfo();

static const Slot channelSlot =
	initKinSynChanCinfo()->getSlot( "channel.channel" );
static const Slot origChannelSlot =
	initKinSynChanCinfo()->getSlot( "origChannel" );
static const Slot gkSlot =
	initKinSynChanCinfo()->getSlot( "GkSrc" );
static const Slot ikSlot =
	initKinSynChanCinfo()->getSlot( "IkSrc" );
static const Slot synapseSlot =
	initKinSynChanCinfo()->getSlot( "synapse" );

/**
   Default constructor sets all the constant values in the expression
*/

KinSynChan::KinSynChan():rInf_(1.0), pulseWidth_(1e-3), tau1_(1e-3)
{
}


void KinSynChan::setRinf(const Conn* conn, double value)
{
    static_cast<KinSynChan*> (conn->data())->innerSetRinf(value);
}

void KinSynChan::innerSetRinf(double value)
{
    rInf_ = value;
}


double KinSynChan::getRinf(Eref e)
{
    return static_cast<KinSynChan*>(e.data())->innerGetRinf();
}

double KinSynChan::innerGetRinf()
{
    return rInf_;
}

//void KinSynChan::innerSetTau2(double value)
//{
//    cout << "Warning: For KinSynChan, tau2 is calculated internally from rInf as: tau2 = (1 - rInf) * tau1." << endl;
//}

//double KinSynChan::innerGetTau2()
//{
//    return tau1_ * (1 - rInf_);
//}

/*
    Decay time constant tau1=1/beta in Destexhe, Mainen and Sejnowski, 1994
*/

double KinSynChan::getTau1(Eref e)
{
    return static_cast<KinSynChan*>(e.data())->innerGetTau1();
}

double KinSynChan::innerGetTau1()
{
    return tau1_;
}
void KinSynChan::setTau1(const Conn* conn, double value)
{
    static_cast<KinSynChan*>(conn->data())->innerSetTau1(value);
}

void KinSynChan::innerSetTau1(double value)
{
    tau1_ = value;
}


/**
   set the Neurotransmitter pulse time for the pre-synaptic part of the channel.
*/

double KinSynChan::getPulseWidth(Eref e)
{
    return static_cast<KinSynChan*>(e.data())->innerGetPulseWidth();
}

double KinSynChan::innerGetPulseWidth()
{
    return pulseWidth_;
}
void KinSynChan::setPulseWidth(const Conn* conn, double value)
{
    static_cast<KinSynChan*>(conn->data())->innerSetPulseWidth(value);
}

void KinSynChan::innerSetPulseWidth(double value)
{
    pulseWidth_ = value;
}

void KinSynChan::innerProcessFunc( Eref e, ProcInfo info )
{
        while ( !pendingEvents_.empty() &&
                (pendingEvents_.top().delay + pulseWidth_) <=  info->currTime_) {
            pendingEvents_.pop();
        }
        if (!pendingEvents_.empty()){
            X_ = X_ * xconst2_ + xconst1_;
        } else {
            X_ = X_ * yconst1_;
        }
	Gk_ = X_ * Gbar_;
	Ik_ = ( Ek_ - Vm_ ) * Gk_;
	send2< double, double >( e, channelSlot, Gk_, Ek_ );
	send2< double, double >( e, origChannelSlot, Gk_, Ek_ );
	send1< double >( e, ikSlot, Ik_ );
	// Usually needed by GHK-type objects
	send1< double >( e, gkSlot, Gk_ );
}

void KinSynChan::innerReinitFunc( Eref e, ProcInfo info )
{
	double dt = info->dt_;
	Gk_ = 0.0;
	Ik_ = 0.0;
	X_ = 0.0;
        tau2_ = (1 - rInf_) * tau1_;
        // xconst1_ and xconst2_ are for the conductance during transmitter pulse.
        // xconst1_ corresponds to A/B * (1 - exp(-B dt)).
        // where A = rInf_ / tau2_ and B = 1/tau2_.
	xconst1_ = rInf_ * ( 1.0 - exp( -dt / tau2_ ) );
	xconst2_ = exp( -dt / tau2_ );
        // yconst1_ is for time when there is no transmitter and the
        // conductance is only decaying.
        // yconst1_ corresponds to exp(-B dt) where B = 1/tau1_.
        yconst1_ = exp( -dt / tau1_);
        
	while ( !pendingEvents_.empty() )
		pendingEvents_.pop();
}
