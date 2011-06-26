// STPNMDAChan.cpp --- 
// 
// Filename: STPNMDAChan.cpp
// Description: 
// Author: Subhasis Ray
// Maintainer: 
// Copyright (C) 2010 Subhasis Ray, all rights reserved.
// Created: Sat Jun 25 15:06:46 2011 (+0530)
// Version: 
// Last-Updated: Sat Jun 25 15:58:56 2011 (+0530)
//           By: Subhasis Ray
//     Update #: 55
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change log:
// 
// 
// 

// Code:

#include "STPNMDAChan.h"

const Cinfo* initSTPNMDAChanCinfo()
{
    static Finfo * STPNMDAChanFinfos[] = {
        new ValueFinfo("MgConc",
                       ValueFtype1<double>::global(),
                       GFCAST(&STPNMDAChan::getMgConc),
                       RFCAST(&STPNMDAChan::setMgConc),
                       "[Mg2+] in extracellular medium."),
        new ValueFinfo("unblocked",
                       ValueFtype1<double>::global(),
                       GFCAST(&STPNMDAChan::getUnblocked),
                       RFCAST(&dummyFunc),
                       "Fraction of channels recovered from Mg2+ block. "
                       "This is an intermediate variable which corresponds to g(V, [Mg2+]o)"
                       " in the equation for conductance:\n"
                       " c * g(V, [Mg2+]o) * S(t) "),
        new ValueFinfo("saturation", ValueFtype1< double >::global(),
                       GFCAST(&STPNMDAChan::getSaturation),
                       RFCAST(&STPNMDAChan::setSaturation),
                       "An upper limit on the NMDA conductance."),
        new LookupFinfo("transitionParam", LookupFtype<double, unsigned int>::global(),
                        GFCAST(&STPNMDAChan::getTransitionParam),
                        RFCAST(&STPNMDAChan::setTransitionParam),
                        "Transition parameters c0 to c10 in the Mg2+ dependent state transitions."),              
    };
    static string doc[] = {
        "Name", "STPNMDAChan",
        "Author", "Subhasis Ray, 2011, NCBS",
        "Description", "STPNMDAChan: Synaptic channel incorporating short term plasticity and"
        " blockage due to Mg2+."
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

static const Cinfo * STPNMDAChanCinfo = initSTPNMDAChanCinfo();

static const Slot channelSlot =
	initSTPNMDAChanCinfo()->getSlot( "channel.channel" );
static const Slot origChannelSlot =
	initSTPNMDAChanCinfo()->getSlot( "origChannel" );
static const Slot gkSlot =
	initSTPNMDAChanCinfo()->getSlot( "GkSrc" );
static const Slot ikSlot =
	initSTPNMDAChanCinfo()->getSlot( "IkSrc" );
static const Slot unblockedSlot =
	initSTPNMDAChanCinfo()->getSlot( "unblockedSrc" );
        
STPNMDAChan::STPNMDAChan():MgConc_(1.5), unblocked_(1.0), saturation_(DBL_MAX)
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
    tau2_ = 0.005;
    tau1_ = 0.130;
    A_ = exp(-c_[8]);
    B1_ = exp(-c_[9]);
    B2_ = exp(-c_[10]);    
}

STPNMDAChan::~STPNMDAChan()
{
    ;
}

void STPNMDAChan::innerProcessFunc(Eref e, ProcInfo proc)
{
    while (!oldEvents_.empty() &&
           oldEvents_.top().delay <= proc->currTime_){
        SynInfo event = oldEvents_.top();
        oldEvents_.pop();
        activation_ -= event.weight / tau2_;
        X_ -= event.weight; 
        Y_ += event.weight;
    }
    while ( !pendingEvents_.empty() &&
            pendingEvents_.top().delay <= proc->currTime_ ) {
        SynInfo event = pendingEvents_.top();
        pendingEvents_.pop();
        activation_ += event.weight / tau2_;
    }
    // TODO: May need to optimize these exponentiations
    double a1_ = exp(-c_[0] * Vm_ - c_[1]);
    double a2_ = 1000.0 * MgConc_ * exp(-c_[2] * Vm_ - c_[3]);
    double b1_ = exp(c_[4] * Vm_ + c_[5] );
    double b2_ = exp(c_[6] * Vm_ + c_[7]);
    // The following two lines calculate next values of x_ and y_
    // according to Forward Euler method:
    // x' = activation
    // y' = -y/tau2
    X_ += activation_ * proc->dt_; 
    Y_ = Y_ * decayFactor_;
    unblocked_ = 1.0 / ( 1.0 + (a1_ + a2_) * (a1_ * B1_ + a2_ * B2_) / (A_ * (a1_ * (B1_ + b1_) + a2_ * (B2_ + b2_))));
    Gk_ = X_ + Y_;
    if (Gk_ > saturation_ * Gbar_){
        Gk_ = saturation_ * Gbar_;
    }
    Gk_ *= unblocked_;
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

void STPNMDAChan::innerReinitFunc(Eref e, ProcInfo proc)
{
    Gk_ = 0.0;
    X_ = 0.0;
    Y_ = 0.0;
    unblocked_ = 1.0;
    activation_ = 0.0;
    modulation_ = 1.0;
    decayFactor_ = exp(-proc->dt_ / tau1_);
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
    dt_tauF_ = proc->dt_ / tauF_;
    dt_tauD1_ = proc->dt_ / tauD1_;
    dt_tauD2_ = proc->dt_ / tauD2_;
}

void STPNMDAChan::innerSynapseFunc(const Conn * conn, double time)
{
    unsigned int index = conn->targetIndex();
    if ( index > synapses_.size() ){
        cout << "STPNMDAChan::innerSynapseFunc:: Index: " << index << ", size: " << synapses_.size() << endl;
        return;
    }
    F_[index] += deltaF_;
    D1_[index] *= d1_;
    D2_[index] *= d2_;    
    if ( mtrand() < amp_[index] * F_[index] * D1_[index] * D2_[index] )
    {
        pendingEvents_.push(synapses_[index].event( time ));
        oldEvents_.push(synapses_[index].event(time + tau2_));
    }
}

void STPNMDAChan::setMgConc(const Conn * conn, double value)
{
    static_cast<STPNMDAChan*>(conn->data())->innerSetMgConc(value);
}

double STPNMDAChan::getMgConc(Eref e)
{
    return static_cast<STPNMDAChan*>(e.data())->innerGetMgConc();
}

void STPNMDAChan::innerSetMgConc(double value)
{
    MgConc_ = value;
}

double STPNMDAChan::innerGetMgConc()
{
    return MgConc_;
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

void STPNMDAChan::setTransitionParam(const Conn* conn, double value, const unsigned int& index )
{
    static_cast< STPNMDAChan* >( conn->data() )->innerSetTransitionParam(conn->target(), value, index);
}

void STPNMDAChan::innerSetTransitionParam(Eref e, double value, const unsigned int index)
{
    if ((index < 0) || (index >= c_.size())){
        cout << "Error: STPNMDAChan::innerSetTransitionParam - index out of range." << endl;
        return;
    }
    c_[index] = value;
}

double STPNMDAChan::getTransitionParam(Eref e, const unsigned int& index)
{
    return static_cast< STPNMDAChan* >(e.data())->innerGetTransitionParam(e, index);
}

/**
   get the transition parameter according to index. See class
   documentation for more information.
*/
double STPNMDAChan::innerGetTransitionParam(Eref e, unsigned int index)
{
    if ((index < 0) || (index >= c_.size())){
        cout << "Error: The index must be between 0 and 10 (inclusive)." << endl;
        return 0.0;
    }
    return c_[index];
}


// 
// STPNMDAChan.cpp ends here
