// NMDAChan.cpp --- 
// 
// Filename: NMDAChan.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Created: Sun Feb 28 18:17:56 2010 (+0530)
// Version: 
// Last-Updated: Mon Jan  7 18:03:58 2013 (+0530)
//           By: subha
//     Update #: 706
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
#include <cfloat>
#include <queue>
#include "gsl/gsl_math.h"
#include "basecode/header.h"
#include "basecode/moose.h"

#include "SynInfo.h"
#include "NMDAChan.h"

double NMDAChan::epsilon_ = (pow(2, 26) - 1) * DBL_EPSILON / 2;
const Cinfo* initNMDAChanCinfo()
{
    ///////////////////////////////////////////////////////
    // Field definitions
    ///////////////////////////////////////////////////////
    
    static Finfo* NMDAChanFinfos[] =
            {
                new LookupFinfo("transitionParam", LookupFtype<double, unsigned int>::global(),
                               GFCAST(&NMDAChan::getTransitionParam),
                               RFCAST(&NMDAChan::setTransitionParam),
                               "Transition parameters c0 to c10 in the Mg2+ dependent state transitions."),
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
                new ValueFinfo("saturation", ValueFtype1< double >::global(),
                               GFCAST(&NMDAChan::getSaturation),
                               RFCAST(&NMDAChan::setSaturation),
                               "An upper limit on the NMDA conductance."),
            
                ///////////////////////////////////////////////////////
                // MsgSrc definitions
                ///////////////////////////////////////////////////////
		new SrcFinfo( "unblockedSrc", Ftype1< double >::global() ),

                ///////////////////////////////////////////////////////
                // MsgDest definitions
                ///////////////////////////////////////////////////////
		new DestFinfo( "MgConcDest", Ftype1< double >::global(),
				RFCAST( &NMDAChan::setMgConc ) ,
				"Update [Mg2+] from other sources at every time step." ),
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
                "S(t) is the ligand gated component of the conductance. It rises "
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
                "After tau2 time, this component is removed from the slope (reduced by weight/tau1) "
                "and added over to the rate of decay of S(t)."
                , 
            };

    static Cinfo NMDAChanCinfo(
            doc,
            sizeof( doc ) / sizeof( string ),		
            initSynChanCinfo(),
            NMDAChanFinfos,
            sizeof( NMDAChanFinfos )/sizeof(Finfo *),
            ValueFtype1< NMDAChan >::global());

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
static const Slot unblockedSlot =
	initNMDAChanCinfo()->getSlot( "unblockedSrc" );


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
                      Mg_(1.5), // mM (value from Traub et al 2005)
                      unblocked_(1.0),
                      saturation_(1.0)
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
void NMDAChan::innerSetTransitionParam(double value, const unsigned int index)
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
   Static function for setting the transition parameter.
*/
void NMDAChan::setTransitionParam(const Conn* conn, double value, const unsigned int& index )
{
    static_cast< NMDAChan* >( conn->data() )->innerSetTransitionParam(value, index);
}

/**
   return the transition parameters c1 - c11 specified by the index.
*/
double NMDAChan::getTransitionParam(Eref e, const unsigned int& index)
{
    return static_cast< NMDAChan* >(e.data())->innerGetTransitionParam(index);
}

/**
   get the transition parameter according to index. See class
   documentation for more information.
*/
double NMDAChan::innerGetTransitionParam(unsigned int index)
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

/**
   get the upper limit on channel conductance
*/
double NMDAChan::getSaturation(Eref e)
{
    return static_cast < NMDAChan* >( e.data() )->innerGetSaturation() ;
}
double NMDAChan::innerGetSaturation()
{
    return saturation_;
}

/**
   Set the upper limit on channel conductance
*/
void NMDAChan::setSaturation(const Conn * conn, double value)
{
    static_cast< NMDAChan* >(conn->data())->innerSetSaturation(value);
}
void NMDAChan::innerSetSaturation(double value)
{
    saturation_ = value;
}

unsigned int NMDAChan::updateNumSynapse( Eref e )
{
    static const Finfo* synFinfo = initNMDAChanCinfo()->findFinfo( "synapse" );
    unsigned int n = e.e->numTargets( synFinfo->msg(), e.i );
    if ( n >= synapses_.size())
        synapses_.resize(n);
    return synapses_.size();
}

void NMDAChan::innerSynapseFunc( const Conn* c, double time )
{
    	unsigned int index = c->targetIndex();
	assert( index < synapses_.size() );
	pendingEvents_.push( synapses_[index].event( time ) );
        oldEvents_.push(synapses_[index].event(time + tau2_));
	// cout << "Event at: " << time << " to " << c->target().name() << " from " << c->source().name() <<  endl;
}

void NMDAChan::processFunc(const Conn* conn, ProcInfo info)
{
    static_cast<NMDAChan*>(conn->data())->innerProcessFunc(conn->target(), info);
}
    
void NMDAChan::innerProcessFunc(Eref e, ProcInfo info)
{
    // cout << "t=" << info->currTime_ << ", " << e.name() << "\n";
    while (!oldEvents_.empty() &&
           gsl_fcmp(oldEvents_.top().delay, info->currTime_, NMDAChan::epsilon_) <= 0){
        // the info->dt_ added above to compensate for off by 1 error for old events.
        SynInfo event = oldEvents_.top();
        oldEvents_.pop();
        activation_ -= event.weight / tau2_;
        // Move the weight from ramp part (X) to decay part (Y).  The
        // second part is for avoiding X < 0 because when activation
        // becomes 0 above, X is still one step behind (weight -
        // activation * dt/ tau).
      
        // If each time step is of length dt, and the ramp length is T,
	// at step n
	// X = n*dt*w/T while n * dt < T .
	// if T <= (n+1)*dt,
	// X = w * (n * dt - T)/T
	// now if n * dt != T, X is never reset to 0 in the older scheme.
        X_ -= event.weight;
        if (X_ < 0){
            X_ = 0.0;
        }
        Y_ += event.weight;
	// cout << "  old: X=" << X_ << ", Y=" << Y_ << ", activation=" << activation_ << endl;
    }
    while ( !pendingEvents_.empty() &&
            gsl_fcmp(pendingEvents_.top().delay, info->currTime_, NMDAChan::epsilon_) <= 0) {
        SynInfo event = pendingEvents_.top();
        pendingEvents_.pop();
        // Activation summates the slopes due to each event
        activation_ += event.weight / tau2_;        
	// cout << "  pending: activation=" << activation_  << " weight=" << event.weight << endl;
    }
    // TODO: May need to optimize these exponentiations
    double a1_ = exp(-c0_ * Vm_ - c1_); // A1_ in traub_nmda.mod
    double a2_ = 1000.0 * Mg_ * exp(-c2_ * Vm_ - c3_); // A2_ in traub_nmda.mod
    double b1_ = exp(c4_ * Vm_ + c5_ ); // B1_ in traub_nmda.mod
    double b2_ = exp(c6_ * Vm_ + c7_); // B2_ in traub_nmda.mod
    // The following two lines calculate next values of X_ and Y_
    // according to Forward Euler method:
    // x' = activation, i.e. X increases as a ramp with slope "activation"
    // y' = -y/tau2, i.e., Y decays exponentially
    X_ += activation_  * info->dt_;
    Y_ = Y_ * decayFactor_;
    // cout << "  integrate:X=" << X_ << ", Y=" << Y_ << ", activation=" << activation_ << endl;
    unblocked_ = 1.0 / ( 1.0 + (a1_ + a2_) * (a1_ * B1_ + a2_ * B2_) / (A_ * (a1_ * (B1_ + b1_) + a2_ * (B2_ + b2_))));
    // cout << "  Mg_unblocked: " << unblocked_ << endl;
    // cout << "  Approx: " << 1/(1+exp(-62*Vm_)*Mg_/3.57) << endl;
    Gk_ = X_ + Y_;
    if (Gk_ > saturation_ * Gbar_){
        Gk_ = saturation_ * Gbar_;
    }
    Gk_  *= unblocked_;
    // cout << "  Gk: " << Gk_ << endl;
    Ik_ = ( Ek_ - Vm_ ) * Gk_;
    // activation_ = 0.0;
    // modulation_ = 1.0;
    send2< double, double >( e, channelSlot, Gk_, Ek_ );
    send2< double, double >( e, origChannelSlot, Gk_, Ek_ );
    send1< double >( e, ikSlot, Ik_ );
    // Usually needed by GHK-type objects
    send1< double >( e, gkSlot, Gk_ );
    send1< double >( e, unblockedSlot, unblocked_);
}

void NMDAChan::reinitFunc(const Conn* conn, ProcInfo info)
{
    static_cast<NMDAChan*>(conn->data())->innerReinitFunc(conn->target(), info);
}

void NMDAChan::innerReinitFunc(Eref e, ProcInfo info)
{
    int exponent;
    double frac = frexp(info->dt_, &exponent);
    // Let us get an epsilon to ignore 1/1000-th differences in values of dt.
    exponent = 52 + exponent - 10; // beware of IEEE double precision specific code
    NMDAChan::epsilon_ = (ldexp(double(1.0), exponent) - 1) * DBL_EPSILON / 2;
    Gk_ = 0.0;
    X_ = 0.0; // A in traub_nmda.mod
    Y_ = 0.0; // B in traub_nmda.mod
    unblocked_ = 1.0; // Mg_unblocked in traub_nmda.mod
    activation_ = 0.0; // equivalent to k in traub_nmda.mod
    // modulation_ = 1.0;
    decayFactor_ = exp(-info->dt_ / tau1_);
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
