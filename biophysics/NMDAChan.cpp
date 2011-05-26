/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** This program is free software; you can redistribute it and/or
** modify it under the terms of the GNU General Public License as
** published by the Free Software Foundation; either version 3, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
** General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program; see the file COPYING.  If not, write to
** the Free Software Foundation, Inc., 51 Franklin Street, Fifth
** Floor, Boston, MA 02110-1301, USA.
**********************************************************************/
// NMDAChan.cpp --- 
// 
// Filename: NMDAChan.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sun Feb 28 18:17:56 2010 (+0530)
// Version: 
// Last-Updated: Fri Oct  8 16:05:10 2010 (+0530)
//           By: Subhasis Ray
//     Update #: 509
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
//  The implementation is for the model by Traub, et al. 2005.
// 

// Change log:
// 
// 2010-10-08 16:04:33 (+0530) - switched tau1 and tau2 to maintain
// compatibility with SynChan : tau1 is decay time constant and tau2
// is rise time constant.
// 
// 
// 
// 

// Code:

/*
#include <cmath>
*/
#include <queue>
#include <cfloat>
#include "header.h"

#include "Synapse.h"
#include "SynBase.h"
#include "ChanBase.h"
#include "SynChanBase.h"
#include "SynChan.h"
#include "NMDAChan.h"

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
static SrcFinfo1< double > unblocked( "unblockedOut", 
	"Sends fraction of unblocked channels"
);

const Cinfo* initNMDAChanCinfo()
{
    ///////////////////////////////////////////////////////
    // Field definitions
    ///////////////////////////////////////////////////////
    
 	static LookupValueFinfo< NMDAChan, unsigned int, double >
		transitionParam(
		"transitionParam", 
		"Transition parameters c0 to c10 in the Mg2+ dependent"
		"state transitions.",
		&NMDAChan::setTransitionParam,
		&NMDAChan::getTransitionParam
	);
	static ValueFinfo< NMDAChan, double > MgConc( "MgConc",
		"External Mg2+ concentration",
		&NMDAChan::setMgConc,
		&NMDAChan::getMgConc
	);

	static ReadOnlyValueFinfo< NMDAChan, double > unblocked("unblocked",
		"Fraction of channels recovered from Mg2+ block. "
		"This is an intermediate variable which corresponds to "
		"g(V, [Mg2+]o) "
		" in the equation for conductance:"
		" c * g(V, [Mg2+]o) * S(t) ",
		&NMDAChan::getUnblocked
	);
	static ValueFinfo< NMDAChan, double > saturation("saturation",
		"An upper limit on the NMDA conductance.",
		&NMDAChan::setSaturation,
		&NMDAChan::getSaturation
	);

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	static DestFinfo MgConcDest( "MgConcDest", 
		"Update [Mg2+] from other sources at every time step.",
		new OpFunc1< NMDAChan, double >( &NMDAChan::setMgConc )
	);
    static Finfo* NMDAChanFinfos[] =
	{
		&unblocked,			// Src
		&transitionParam,	// LookupValue
		&MgConc,			// Value
		&unblocked,			// ReadOnlyValue
		&saturation,		// Value
		&MgConcDest,			// Dest
	};


    static string doc[] =
            {
		"Name", "NMDAChan",
		"Author", "Subhasis Ray, 2010, NCBS",
		"Description", "NMDAChan: Extracellular [Mg2+] dependent NMDA channel."
                "This channel has four states as described by Jahr and Stevens (J. Neurosci. 1990, 10(9)) "
                "This implementation is based on equation 4(a) in that article. "
                "The channel conductance is defined as :"
                " c * g(V, [Mg2+]o) * S(t) "
                "where c is a scaling constant. "
                "S(t) is the legand gated component of the conductance. It rises "
                "linearly for t = tau2. Then decays exponentially with time constant "
                "t = tau1. "
                "g is a function of voltage and the extracellular [Mg2+] defined as: "
                "1 / { 1 + (a1 + a2) * (a1 * B1 + a2 * B2)/ [A * a1 * (b1 + B1) + A * a2 * (b2 + B2)]} "
                "a1 = 1e3 * exp( - c0 * V - c1) s^-1, c0 = 16.0 / V, c1 = 2.91 "
                "a2 = 1e-3 * [Mg2+] * exp( -c2 * V - c3) mM^-1 s, c2 = 45.0 / V, c3 = 6.97 "
                "b1 = 1e3 * exp(c4  * V + c5) s^-1, c4 = 9.0 / V, c5 = 1.22 "
                "b2 = 1e3 * exp(c6 * V + c7) s^-1, c6 = 17.0 / V, c7 = 0.96 "
                "A = 1e3 * exp(-c8) s^-1, c8 = 2.847 "
                "B1 = 1e3 * exp(-c9) s^-1, c9 = 0.693 s^-1 "
                "B2 = 1e3 * exp(-c10) s^-1, c10 = 3.101. "
                "The behaviour of S(t) is as follows: "
                "If a spike arrives, then the slope of the linear rise of S(t) is incremented by weight / tau2. "
                "After tau2 time, this component is removed from the slope (reduced by weight/tau) "
                "and added over to the rate of decay of S(t)."
                , 
            };

    static Cinfo NMDAChanCinfo(
		"NMDAChan",
		SynChan::initCinfo(),
		NMDAChanFinfos, sizeof( NMDAChanFinfos )/sizeof(Finfo *),
		new Dinfo< NMDAChan >()
	);

    return &NMDAChanCinfo;
}


static const Cinfo* NMDAChanCinfo = NMDAChan::initCinfo();

/**
   Default constructor sets all the constant values in the expression
   for [Mg2+] dependent component of the channel conductance according
   to Jahr and Stevens (Sept. 1990) equation 4(a) for details.
*/
NMDAChan::NMDAChan(): c0_(16.0),
                      c1_(2.91),
                      c2_(45.0),
                      c3_(6.97),
                      c4_(9.0),
                      c5_(1.22),
                      c6_(17.0),
                      c7_(0.96),
                      c8_(2.847),
                      c9_(0.693),
                      c10_(3.101),
                      x_(0.0),
                      y_(0.0),
                      Mg_(1.5), // mM (value from Traub et al 2005)
                      unblocked_(0.0),
                      saturation_(DBL_MAX)
{
    tau2_ = 0.005;
    tau1_ = 0.130;
    A_ = exp(-c8_);
    B1_ = exp(-c9_);
    B2_ = exp(-c10_);
}


/**
   Set one of the the constants c0 - c10 in the expression by Jahr and
   Stevens.
*/
void NMDAChan::setTransitionParam(unsigned int index, double value)
{
    switch (index) {
        case 0: c0_ = value; break;
        case 1: c1_ = value; break;
        case 2: c2_ = value; break;
        case 3: c3_ = value; break;
        case 4: c4_ = value; break;
        case 5: c5_ = value; break;
        case 6: c6_ = value; break;
        case 7: c7_ = value; break;
        case 8: c8_ = value; A_ = 1e3 * exp(-c8_); break;
        case 9: c9_ = value; B1_ = 1e3 * exp(-c9_); break;
        case 10: c10_ = value; B2_ = 1e3 * exp(-c10_); break;
        default: cout << "Error: The index must be between 0 and 10 (inclusive)." << endl;
    }
}

/**
   get the transition parameter according to index. See class
   documentation for more information.
*/
double NMDAChan::getTransitionParam( unsigned int index ) const
{
    switch (index) {
        case 0: return c0_;
        case 1: return c1_;
        case 2: return c2_;
        case 3: return c3_;
        case 4: return c4_;
        case 5: return c5_;
        case 6: return c6_;
        case 7: return c7_;
        case 8: return c8_;
        case 9: return c9_;
        case 10: return c10_;
        default: cout << "Error: The index must be between 0 and 10 (inclusive)." << endl;
    }
    return 0.0;
}

/**
   set the [Mg2+] for the channel.
*/

void NMDAChan::setMgConc(double value)
{
    Mg_ = value;
}

double NMDAChan::getMgConc() const
{
    return Mg_;
}

double NMDAChan::getUnblocked() const
{
    return unblocked_;
}

/**
   get the upper limit on channel conductance
*/
double NMDAChan::getSaturation() const
{
    return saturation_;
}

/**
   Set the upper limit on channel conductance
*/
void NMDAChan::setSaturation(double value)
{
    saturation_ = value;
}

/*
unsigned int NMDAChan::updateNumSynapse( Eref e )
{
    static const Finfo* synFinfo = initNMDAChanCinfo()->findFinfo( "synapse" );
    unsigned int n = e.e->numTargets( synFinfo->msg(), e.i );
    if ( n >= synapses_.size())
        synapses_.resize(n);
    return synapses_.size();
}
*/
void NMDAChan::process( const Eref& e, ProcPtr info )
{
    while (!oldEvents_.empty() &&
           oldEvents_.top().getDelay() <= info->currTime){
        Synapse event = oldEvents_.top();
        oldEvents_.pop();
        activation_ -= event.getWeight() / tau2_;
        x_ -= event.getWeight(); 
        y_ += event.getWeight();
    }
    while ( !pendingEvents_.empty() &&
            pendingEvents_.top().getDelay() <= info->currTime ) {
        Synapse event = pendingEvents_.top();
        pendingEvents_.pop();
        activation_ += event.getWeight() / tau2_;
        // oldEvents_.push(event.event(tau2_));
        oldEvents_.push( Synapse( event, tau2_ ) );
    }
    // TODO: May need to optimize these exponentiations
	double Vm = getVm();

    double a1_ = exp(-c0_ * Vm - c1_);
    double a2_ = 1000.0 * Mg_ * exp(-c2_ * Vm - c3_);
    double b1_ = exp(c4_ * Vm + c5_ );
    double b2_ = exp(c6_ * Vm + c7_);
    // The following two lines calculate next values of x_ and y_
    // according to Forward Euler method:
    // x' = activation
    // y' = -y/tau2
    x_ += activation_ * info->dt; 
    y_ = y_ * decayFactor_;
    unblocked_ = 1.0 / ( 1.0 + (a1_ + a2_) * (a1_ * B1_ + a2_ * B2_) / (A_ * (a1_ * (B1_ + b1_) + a2_ * (B2_ + b2_))));
    double Gk = getGbar() * (x_ + y_) * unblocked_;
    if (Gk > saturation_){
        Gk = saturation_;
    }
	setGk( Gk );
	updateIk();
    // Ik_ = ( Ek_ - Vm_ ) * Gk_;

    modulation_ = 1.0;

	SynChanBase::process( e, info ); // Sends out channel messages.
	/*
    send2< double, double >( e, channelSlot, Gk_, Ek_ );
    send2< double, double >( e, origChannelSlot, Gk_, Ek_ );
    send1< double >( e, ikSlot, Ik_ );
    // Usually needed by GHK-type objects
    send1< double >( e, gkSlot, Gk_ );
	*/
	unblocked.send( e, info, unblocked_ );
    // send1< double >( e, unblockedSlot, unblocked_);
}

void NMDAChan::reinit( const Eref& e, ProcPtr info)
{
	setGk( 0.0 );
    x_ = 0.0;
    y_ = 0.0;
    unblocked_ = 0.0;
    activation_ = 0.0;
    modulation_ = 1.0;
    decayFactor_ = exp(-info->dt / tau1_);
	setIk( 0.0 );
    // updateNumSynapse( e );
    while(!pendingEvents_.empty()){
        pendingEvents_.pop();
    }
    
    while(!oldEvents_.empty()){
        oldEvents_.pop();
    }
}

// 
// nmdachan.cpp ends here
