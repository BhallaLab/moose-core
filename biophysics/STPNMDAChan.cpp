/*******************************************************************
 * File:            STPNMDAChan.cpp
 * Description:      
 * Author:          Subhasis Ray
 * 
 * Created:         2011-06-15 14:45:52 (+0530)
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

#ifndef _STPNMDACHAN_CPP
#define _STPNMDACHAN_CPP
#include <math.h>
#include "moose.h"
#include <queue>
#include "randnum/randnum.h"

#include "SynInfo.h"
#include "SynChan.h"
#include "STPSynChan.h"
#include "STPNMDAChan.h"

extern const Cinfo * initSTPSynChanCinfo();

static const double SynE = 2.7182818284590452354;

const Cinfo* initSTPNMDAChanCinfo()
{
    ///////////////////////////////////////////////////////
    // Field definitions
    ///////////////////////////////////////////////////////

    static Finfo* STPNMDAChanFinfos[] =
            {
                new LookupFinfo("transitionParam", LookupFtype<double, unsigned int>::global(),
                               GFCAST(&STPNMDAChan::getTransitionParam),
                               RFCAST(&STPNMDAChan::setTransitionParam),
                               "Transition parameters c0 to c10 in the Mg2+ dependent state transitions."),
                new ValueFinfo("MgConc", ValueFtype1< double >::global(),
                               GFCAST(&STPNMDAChan::getMgConc),
                               RFCAST(&STPNMDAChan::setMgConc),
                               "External Mg2+ concentration"),
                new ValueFinfo("unblocked", ValueFtype1< double >::global(),
                               GFCAST(&STPNMDAChan::getUnblocked),
                               RFCAST(&dummyFunc),
                               "Fraction of channels recovered from Mg2+ block. "
                               "This is an intermediate variable which corresponds to g(V, [Mg2+]o) "
                               " in the equation for conductance:\n "
                               " c * g(V, [Mg2+]o) * S(t) "
                               ),
                new ValueFinfo("saturation", ValueFtype1< double >::global(),
                               GFCAST(&STPNMDAChan::getSaturation),
                               RFCAST(&STPNMDAChan::setSaturation),
                               "An upper limit on the NMDA conductance."),
            
                ///////////////////////////////////////////////////////
                // MsgSrc definitions
                ///////////////////////////////////////////////////////
		new SrcFinfo( "unblockedSrc", Ftype1< double >::global() ),

                ///////////////////////////////////////////////////////
                // MsgDest definitions
                ///////////////////////////////////////////////////////
		new DestFinfo( "MgConcDest", Ftype1< double >::global(),
				RFCAST( &STPNMDAChan::setMgConc ) ,
				"Update [Mg2+] from other sources at every time step." ),
            };

    static string doc[] =
            {
		"Name", "STPNMDAChan",
		"Author", "Subhasis Ray, 2011, NCBS",
		"Description", "STPNMDAChan: Synaptic channel incorporating short term plasticity and"
                " blockage due to [Mg2+]. This is a combination of STPSynChan and"
                " NMDAChan."
            };
    static Cinfo STPNMDAChanCinfo(
            doc,
            sizeof( doc ) / sizeof( string ),		
            initSTPSynChanCinfo(),
            STPNMDAChanFinfos,
            sizeof( STPNMDAChanFinfos )/sizeof(Finfo *),
            ValueFtype1< STPNMDAChan >::global());

    return &STPNMDAChanCinfo;
}

static const Cinfo* STPNMDAChanCinfo = initSTPNMDAChanCinfo();

static const Slot channelSlot =
	initSTPNMDAChanCinfo()->getSlot( "channel" );
static const Slot origChannelSlot =
	initSTPNMDAChanCinfo()->getSlot( "origChannel" );
static const Slot gkSlot =
	initSTPNMDAChanCinfo()->getSlot( "GkSrc" );
static const Slot ikSlot =
	initSTPNMDAChanCinfo()->getSlot( "IkSrc" );
static const Slot synapseSlot =
	initSTPNMDAChanCinfo()->getSlot( "synapse" );
static const Slot unblockedSlot =
	initSTPNMDAChanCinfo()->getSlot( "unblockedSrc" );


STPNMDAChan::STPNMDAChan()
{
    c_.resize(11);
    c_[0] = 16.0;
    c_[1] = 2.91;
    c_[2] = 45.0;
    c_[3] = 6.97;
    c_[4] = 9.0;
    c_[5] = 1.22;
    c_[6] = 17.0;
    c_[7] = 0.96;
    c_[8] = 2.847;
    c_[9] = 0.693;
    c_[10] = 3.101;
    Mg_ = 1.5; // mM  = value from Traub et al 2005
    unblocked_ = 0.0;
    saturation_ = DBL_MAX;    
    tau2_ = 0.005;
    tau1_ = 0.130;
    A_ = exp(-c_[8]);
    B1_ = exp(-c_[9]);
    B2_ = exp(-c_[10]);    
}


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
void STPNMDAChan::innerSetTransitionParam(double value, const unsigned int index)
{
    if ((index < 0) || (index >= c_.size())){
        cout << "Error: STPNMDAChan::innerSetTransitionParam - index out of range." << endl;
        return;
    }
    c_[index] = value;
}

void STPNMDAChan::setTransitionParam(const Conn* conn, double value, const unsigned int& index )
{
    static_cast< STPNMDAChan* >( conn->data() )->innerSetTransitionParam(value, index);
}

double STPNMDAChan::getTransitionParam(Eref e, const unsigned int& index)
{
    return static_cast< STPNMDAChan* >(e.data())->innerGetTransitionParam(index);
}

/**
   get the transition parameter according to index. See class
   documentation for more information.
*/
double STPNMDAChan::innerGetTransitionParam(unsigned int index)
{
    if ((index < 0) || (index >= c_.size())){
        cout << "Error: The index must be between 0 and 10 (inclusive)." << endl;
        return 0.0;
    }
    return c_[index];
}

void STPNMDAChan::setMgConc(const Conn* conn, double value)
{
    static_cast<STPNMDAChan*> (conn->data())->innerSetMgConc(value);
}

void STPNMDAChan::innerSetMgConc(double value)
{
    Mg_ = value;
}

double STPNMDAChan::getMgConc(Eref e)
{
    return static_cast<STPNMDAChan*>(e.data())->innerGetMgConc();
}

double STPNMDAChan::innerGetMgConc()
{
    return Mg_;
}

/**
   get the fraction of unblocked channels
*/
double STPNMDAChan::getUnblocked(Eref e)
{
    return static_cast<STPNMDAChan*>(e.data())->innerGetUnblocked();
}

double STPNMDAChan::innerGetUnblocked()
{
    return unblocked_;
}

/**
   get the upper limit on channel conductance
*/
double STPNMDAChan::getSaturation(Eref e)
{
    return static_cast < STPNMDAChan* >( e.data() )->innerGetSaturation() ;
}

double STPNMDAChan::innerGetSaturation()
{
    return saturation_;
}

/**
   Set the upper limit on channel conductance
*/
void STPNMDAChan::setSaturation(const Conn * conn, double value)
{
    static_cast< STPNMDAChan* >(conn->data())->innerSetSaturation(value);
}

void STPNMDAChan::innerSetSaturation(double value)
{
    saturation_ = value;
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void STPNMDAChan::innerProcessFunc(Eref e, ProcInfo info)
{
    while (!oldEvents_.empty() &&
           oldEvents_.top().delay <= info->currTime_){
        SynInfo event = oldEvents_.top();
        oldEvents_.pop();
        activation_ -= event.weight / tau2_;
        X_ -= event.weight; 
        Y_ += event.weight;
    }
    while ( !pendingEvents_.empty() &&
            pendingEvents_.top().delay <= info->currTime_ ) {
        SynInfo event = pendingEvents_.top();
        pendingEvents_.pop();
        activation_ += event.weight / tau2_;
        oldEvents_.push(event.event(tau2_));
    }
    // TODO: May need to optimize these exponentiations
    double a1_ = exp(-c_[0] * Vm_ - c_[1]);
    double a2_ = 1000.0 * Mg_ * exp(-c_[2] * Vm_ - c_[3]);
    double b1_ = exp(c_[4] * Vm_ + c_[5] );
    double b2_ = exp(c_[6] * Vm_ + c_[7]);
    // The following two lines calculate next values of x_ and y_
    // according to Forward Euler method:
    // x' = activation
    // y' = -y/tau2
    X_ += activation_ * info->dt_; 
    Y_ = Y_ * decayFactor_;
    unblocked_ = 1.0 / ( 1.0 + (a1_ + a2_) * (a1_ * B1_ + a2_ * B2_) / (A_ * (a1_ * (B1_ + b1_) + a2_ * (B2_ + b2_))));
    Gk_ = Gbar_* (X_ + Y_) * unblocked_;
    if (Gk_ > saturation_){
        Gk_ = saturation_;
    }
    Ik_ = ( Ek_ - Vm_ ) * Gk_;
    modulation_ = 1.0;
    for (unsigned int ii = 0; ii < synapses_.size(); ++ii){
        F_[ii] += (1 - F_[ii]) * dt_tauF_;
        D1_[ii] += (1 - D1_[ii]) * dt_tauD1_;
        D2_[ii] += (1 - D2_[ii]) * dt_tauD2_;
    }

    send2< double, double >( e, channelSlot, Gk_, Ek_ );
    send2< double, double >( e, origChannelSlot, Gk_, Ek_ );
    send1< double >( e, ikSlot, Ik_ );
    send1< double >( e, gkSlot, Gk_ );
    send1< double >( e, unblockedSlot, unblocked_);
}


void STPNMDAChan::innerReinitFunc( Eref e, ProcInfo info )
{
    Gk_ = 0.0;
    X_ = 0.0;
    Y_ = 0.0;
    unblocked_ = 0.0;
    activation_ = 0.0;
    modulation_ = 1.0;
    decayFactor_ = exp(-info->dt_ / tau1_);
    Ik_ = 0.0;
    updateNumSynapse( e );
    while(!pendingEvents_.empty()){
        pendingEvents_.pop();
    }
    
    while(!oldEvents_.empty()){
        oldEvents_.pop();
    }
    
    for (unsigned int index = 0; index < synapses_.size(); ++index){
        F_[index] = initF_[index];
        D1_[index] = initD1_[index];
        D2_[index] = initD2_[index];
        amp_[index] = initPr_[index] / ((initF_[index] + deltaF_) * initD1_[index] * d1_ * initD2_[index] * d2_);
    }
    dt_tauF_ = info->dt_ / tauF_;
    dt_tauD1_ = info->dt_ / tauD1_;
    dt_tauD2_ = info->dt_ / tauD2_;
}
    
void STPNMDAChan::innerSynapseFunc( const Conn* c, double time )
{
    unsigned int index = c->targetIndex();
    assert( index < synapses_.size() );
    F_[index] += deltaF_;
    D1_[index] *= d1_;
    D2_[index] *= d2_;    
    if ( mtrand() < amp_[index] * F_[index] * D1_[index] * D2_[index] )
    {
        pendingEvents_.push(synapses_[index].event( time ));
    }
}


#endif
