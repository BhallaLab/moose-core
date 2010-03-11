// NMDAChan.cpp --- 
// 
// Filename: NMDAChan.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sun Feb 28 18:17:56 2010 (+0530)
// Version: 
// Last-Updated: Thu Mar 11 14:41:40 2010 (+0530)
//           By: Subhasis Ray
//     Update #: 369
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
// 
// 
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
// 
// 

// Code:

#include <cmath>
#include <queue>
#include "basecode/header.h"
#include "basecode/moose.h"

#include "SynInfo.h"
#include "NMDAChan.h"

const Cinfo* initNMDAChanCinfo()
{
    static Finfo* processShared[] =
            {
                new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                               RFCAST( &NMDAChan::processFunc ) ),
                new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                               RFCAST( &NMDAChan::reinitFunc ) ),
            };
    static Finfo* process =	new SharedFinfo( "process", processShared, 
                                                 sizeof( processShared ) / sizeof( Finfo* ),
                                                 "This is a shared message to receive Process message from the scheduler." );
    static Finfo* channelShared[] =
            {
                new SrcFinfo( "channel", Ftype2< double, double >::global() ),
                new DestFinfo( "Vm", Ftype1< double >::global(), 
                               RFCAST( &NMDAChan::channelFunc ) ),
            };

    ///////////////////////////////////////////////////////
    // Field definitions
    ///////////////////////////////////////////////////////

    static Finfo* NMDAChanFinfos[] =
            {
                new ValueFinfo("transitionParam", LookupFtype<int, double>::global(),
                               GFCAST(&NMDAChan::getTransitionParam),
                               RFCAST(&NMDAChan::setTransitionParam),
                               "Transition parameters c1 to c11 in the Mg2+ dependent state transitions."),
                new ValueFinfo("MgConc", ValueFtype1< double >::global(),
                               GFCAST(&NMDAChan::getMgConc),
                               RFCAST(&NMDAChan::setMgConc),
                               "External Mg2+ concentration"),
                new ValueFinfo("unblocked", ValueFtype1< double >::global(),
                               GFCAST(&NMDAChan::getUnblocked),
                               RFCAST(&dummyFunc),
                               "Fraction of channels recovered from Mg2+ block. "
                               "This is an intermediate variable which corresponds to g(V, [Mg2+]o) "
                               " in the equation for conductance:\n "
                               " c * g(V, [Mg2+]o) * S(t) "
                               ),
            
                // The commented out fields are already inherited from SynChan
		// new ValueFinfo( "Gbar", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getGbar ), 
		// 	RFCAST( &NMDAChan::setGbar )
		// ),
		// new ValueFinfo( "Ek", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getEk ), 
		// 	RFCAST( &NMDAChan::setEk )
		// ),
		// new ValueFinfo( "tau1", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getTau1 ), 
		// 	RFCAST( &NMDAChan::setTau1 )
		// ),
		// new ValueFinfo( "tau2", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getTau2 ), 
		// 	RFCAST( &NMDAChan::setTau2 )
		// ),
		// new ValueFinfo( "normalizeWeights", 
		// 	ValueFtype1< bool >::global(),
		// 	GFCAST( &NMDAChan::getNormalizeWeights ), 
		// 	RFCAST( &NMDAChan::setNormalizeWeights )
		// ),
		// new ValueFinfo( "Gk", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getGk ), 
		// 	RFCAST( &NMDAChan::setGk )
		// ),
		// new ValueFinfo( "Ik", ValueFtype1< double >::global(),
		// 	GFCAST( &NMDAChan::getIk ), 
		// 	&dummyFunc
		// ),

		// new ValueFinfo( "numSynapses",
		// 	ValueFtype1< unsigned int >::global(),
		// 	GFCAST( &NMDAChan::getNumSynapses ), 
		// 	&dummyFunc // Prohibit reassignment of this index.
		// ),

		// new LookupFinfo( "weight",
		// 	LookupFtype< double, unsigned int >::global(),
		// 	GFCAST( &NMDAChan::getWeight ),
		// 	RFCAST( &NMDAChan::setWeight )
		// ),

		// new LookupFinfo( "delay",
		// 	LookupFtype< double, unsigned int >::global(),
		// 	GFCAST( &NMDAChan::getDelay ),
		// 	RFCAST( &NMDAChan::setDelay )
		// ),
                ///////////////////////////////////////////////////////
                // Shared message definitions
                ///////////////////////////////////////////////////////
		// process,
		// new SharedFinfo( "process", processShared,
		// 	sizeof( processShared ) / sizeof( Finfo* ),
		// 	"This is a shared message to receive Process message from the scheduler." ), 
		// new SharedFinfo( "channel", channelShared,
		// 	sizeof( channelShared ) / sizeof( Finfo* ),
		// 	"This is a shared message to couple channel to compartment. "
		// 	"The first entry is a MsgSrc to send Gk and Ek to the compartment "
		// 	"The second entry is a MsgDest for Vm from the compartment." ),

                ///////////////////////////////////////////////////////
                // MsgSrc definitions
                ///////////////////////////////////////////////////////
		// new SrcFinfo( "IkSrc", Ftype1< double >::global() ),
		// new SrcFinfo( "GkSrc", Ftype1< double >::global() ),
		// new SrcFinfo( "origChannel", Ftype2< double, double >::
		// 	global() ),

                ///////////////////////////////////////////////////////
                // MsgDest definitions
                ///////////////////////////////////////////////////////
		// new DestFinfo( "synapse", Ftype1< double >::global(),
		// 		RFCAST( &NMDAChan::synapseFunc ) ,
		// 		"Arrival of a spike. Arg is time of sending of spike." ),
		// new DestFinfo( "activation", Ftype1< double >::global(),
		// 		RFCAST( &NMDAChan::activationFunc ),
		// 		"Sometimes we want to continuously activate the channel" ),
		// new DestFinfo( "modulator", Ftype1< double >::global(),
		// 		RFCAST( &NMDAChan::modulatorFunc ),
		// 		"Modulate channel response" ),
            };

    // NMDAChan is scheduled after the compartment calculations.
    static SchedInfo schedInfo[] = { { process, 0, 1 } };

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
                "linearly for t = tau1. Then decays exponentially with time constant "
                "t = tau2. "
                "g is a function of voltage and the extracellular [Mg2+] defined as: "
                "1 / { 1 + (a1 + a2) * (a1 * B1 + a2 * B2)/ [A * a1 * (b1 + B1) + A * a2 * (b2 + B2)]} "
                "a1 = 1e3 * exp( - c0 * V - c1) s^-1, c0 = - 16.0 / V, c1 = 2.91 "
                "a2 = 1e-3 * [Mg2+] * exp( -c2 * V - c3) mM^-1 s, c2 = 45.0 / V, c3 = 6.97 "
                "b1 = 1e3 * exp(c4  * V + c5) s^-1, c4 = 9.0 / V, c5 = 1.22 "
                "b2 = 1e3 * exp(c6 * V + c7) s^-1, c6 = 17.0 / V, c7 = 0.96 "
                "A = 1e3 * exp(-c8) s^-1, c8 = 2.847 "
                "B1 = 1e3 * exp(-c9) s^-1, c9 = 0.693 s^-1 "
                "B2 = 1e3 * exp(-c10) s^-1, c10 = 3.101. "
                "The behaviour of S(t) is as follows: "
                "If a spike arrives, then the slope of the linear rise of S(t) is incremented by weight / tau1. "
                "After tau1 time, this component is removed from the slope (reduced by weight/tau) "
                "and added over to the rate of decay of S(t)."
                , 
            };

    static Cinfo NMDAChanCinfo(
            doc,
            sizeof( doc ) / sizeof( string ),		
            initSynChanCinfo(),
            NMDAChanFinfos,
            sizeof( NMDAChanFinfos )/sizeof(Finfo *),
            ValueFtype1< NMDAChan >::global(),
            schedInfo, 1
                               );

    return &NMDAChanCinfo;
}


static const Cinfo* NMDAChanCinfo = initNMDAChanCinfo();

static const Slot channelSlot =
	initNMDAChanCinfo()->getSlot( "channel.channel" );
static const Slot origChannelSlot =
	initNMDAChanCinfo()->getSlot( "origChannel" );
static const Slot gkSlot =
	initNMDAChanCinfo()->getSlot( "GkSrc" );
static const Slot ikSlot =
	initNMDAChanCinfo()->getSlot( "IkSrc" );
static const Slot NMDAapseSlot =
	initNMDAChanCinfo()->getSlot( "NMDAapse" );


/**
   Default constructor sets all the constant values in the expression
   for [Mg2+] dependent component of the channel conductance according
   to Jahr and Stevens (Sept. 1990) equation 4(a) for details.
*/
NMDAChan::NMDAChan(): c0_(-16.0),
                      c1_(2.91),
                      c2_(45.0),
                      c3_(6.97),
                      c4_(9.0),
                      c5_(1.22),
                      c6_(17.0),
                      c7_(0.96),
                      c8_(2.847),
                      c9_(0.693),
                      c10_(3.101)
{
    A_ = 1e3 * exp(-c8_);
    B1_ = 1e3 * exp(-c9_);
    B2_ = 1e3 * exp(-c10_);
}

#if 0
/**
   Set the constants c1 - c11 in the expression by Jahr and Stevens
   This form expects a vector with all the constants in it.
*/
void NMDAChan::innerTransitionParams(vector<couble> constList)
{
    if (constList.size() != 11){
        cout << "Error: Expect a vector of 11 constants. Got only " << constList.size() << endl;
        return;
    }
    c0_ = constList[0];
    c1_ = constList[1];
    c3_ = constList[2];
    c4_ = constList[3];
    c5_ = constList[4];
    c6_ = constList[5];
    c7_ = constList[6];
    c8_ = constList[7];
    c9_ = constList[8];
    c10_ = constList[9];
    c11_ = constList[10];
    A_ = 1e3 * exp(-c8);
    B1_ = 1e3 * exp(-c9);
    B2_ = 1e3 * exp(-c10);
}

void NMDAChan::setTransitionParams(const Conn* conn, vector<double> constList)
{
    static_cast< NMDAChan* >( c->data() )->innerSetConstants(constList);
}

#endif

/**
   Set one of the the constants c1 - c11 in the expression by Jahr and
   Stevens. The indexing starts at 0.
*/
void NMDAChan::innerSetTransitionParam(int index, double value)
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
        default: cout << "Error: The index must be between 1 and 11 (inclusive)." << endl;
    }
}

/**
   Static function for setting the transition parameter.
*/
void NMDAChan::setTransitionParam(const Conn* conn, int index, double value)
{
    static_cast< NMDAChan* >( conn->data() )->innerSetTransitionParam(index, value);
}

/**
   return the transition parameters c1 - c11 specified by the index.
*/
double NMDAChan::getTransitionParam(Eref e, int index)
{
    return static_cast< NMDAChan* >(e.data())->innerGetTransitionParam(index);
}

/**
   get the transition parameter according to index. See class
   documentation for more information.
*/
double NMDAChan::innerGetTransitionParam(int index)
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
        default: cout << "Error: The index must be between 1 and 11 (inclusive)." << endl;
    }
    return 0.0;
}

/**
   set the [Mg2+] for the channel.
*/

void NMDAChan::setMgConc(const Conn* conn, double value)
{
    static_cast<NMDAChan*> (conn->data())->innerSetMgConc(value);
}

void NMDAChan::innerSetMgConc(double value)
{
    Mg_ = value;
}

double NMDAChan::getMgConc(Eref e)
{
    return static_cast<NMDAChan*>(e.data())->innerGetMgConc();
}

double NMDAChan::innerGetMgConc()
{
    return Mg_;
}

/**
   get the fraction of unblocked channels
*/
double NMDAChan::getUnblocked(Eref e)
{
    return static_cast<NMDAChan*>(e.data())->innerGetUnblocked();
}

double NMDAChan::innerGetUnblocked()
{
    return unblocked_;
}

void NMDAChan::processFunc(const Conn* conn, ProcInfo info)
{
    static_cast<NMDAChan*>(conn->data())->innerProcessFunc(conn->target(), info);
}

void NMDAChan::innerProcessFunc(Eref e, ProcInfo info)
{
    while ( !pendingEvents_.empty() &&
            pendingEvents_.top().delay <= info->currTime_ ) {
        SynInfo event = pendingEvents_.top();
        pendingEvents_.pop();
        activation_ += event.weight / info->dt_;
        oldEvents_.push(event.event(tau1_));
    }
    while (!oldEvents_.empty() &&
           oldEvents_.top().delay <= info->currTime_){
        SynInfo event = oldEvents_.top();
        oldEvents_.pop();
        activation_ -= event.weight / info->dt_;
        // x_ -= event.weight; // this is already incorporated into activation_
        y_ += event.weight;
    }
    // TODO: May need to optimize these exponentiations
    double a1_ = exp(-c0_ * Vm_ - c1_);
    double a2_ = 1e-3 * Mg_ * exp(-c3_ * Vm_ - c3_);
    double b1_ = 1e3 * exp(c4_ * Vm_ + c5_ );
    double b2_ = 1e3 * exp(c6_ * Vm_ + c7_);
    // The following two lines calculate next values of x_ and y_
    // according to Forward Euler method:
    // x' = activation
    // y' = -y/tau2
    x_ += activation_ * info->dt_; 
    y_ += -y_ * info->dt_ / tau2_;
    unblocked_ = 1.0 / ( 1.0 + (a1_ + a2_) * (a1_ * b1_ + a2_ * b2_) / (A_ * (a1_ * (B1_ + b1_) + a2_ * (B2_ + b2_))));
    Gk_ = (x_ + y_) * unblocked_ * (Vm_ - Ek_);
    Ik_ = ( Ek_ - Vm_ ) * Gk_;
    activation_ = 0.0;
    modulation_ = 1.0;
    send2< double, double >( e, channelSlot, Gk_, Ek_ );
    send2< double, double >( e, origChannelSlot, Gk_, Ek_ );
    send1< double >( e, ikSlot, Ik_ );
    // Usually needed by GHK-type objects
    send1< double >( e, gkSlot, Gk_ );    
}

void NMDAChan::reinitFunc(const Conn* conn, ProcInfo info)
{
    static_cast<NMDAChan*>(conn->data())->innerReinitFunc(conn->target(), info);
}

void NMDAChan::innerReinitFunc(Eref e, ProcInfo info)
{
    Gk_ = 0.0;
    x_ = 0.0;
    y_ = 0.0;
    unblocked_ = 0.0;
    activation_ = 0.0;
    modulation_ = 1.0;
    Ik_ = 0.0;
    updateNumSynapse( e );
    while(!pendingEvents_.empty()){
        pendingEvents_.pop();
    }
    
    while(!oldEvents_.empty()){
        oldEvents_.pop();
    }
}

// 
// nmdachan.cpp ends here
