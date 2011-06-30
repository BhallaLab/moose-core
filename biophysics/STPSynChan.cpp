/*******************************************************************
 * File:            StochSynChan.cpp
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-26 10:56:55
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

#ifndef _STPSYNCHAN_CPP
#define _STPSYNCHAN_CPP
#include <math.h>
#include "moose.h"
#include <queue>
#include "randnum/randnum.h"

#include "SynInfo.h"
#include "SynChan.h"
#include "STPSynChan.h"


const Cinfo* initSTPSynChanCinfo()
{
    ///////////////////////////////////////////////////////
    // Field definitions
    ///////////////////////////////////////////////////////

    static Finfo* STPSynChanFinfos[] =
            {
                new LookupFinfo( "initPr",
                                 LookupFtype< double, unsigned int>::global(),
                                 GFCAST( &STPSynChan::getInitPr ),
                                 RFCAST( &STPSynChan::setInitPr ),
                                 "Initial release probability."
                                 ),
                new LookupFinfo( "Pr",
                                 LookupFtype< double, unsigned int>::global(),
                                 GFCAST( &STPSynChan::getPr ),
                                 RFCAST( &dummyFunc ),
                                 "Release probability."
                                 ),
                new ValueFinfo("tauD1",
                               ValueFtype1< double >::global(),
                               GFCAST( &STPSynChan::getTauD1),
                               RFCAST( &STPSynChan::setTauD1),
                               "Time constant for fast depression component. This is common for all"
                               " synapses on this channel."
                               ),
                new ValueFinfo("tauD2",
                               ValueFtype1< double >::global(),
                               GFCAST( &STPSynChan::getTauD2),
                               RFCAST( &STPSynChan::setTauD2),
                               "Time constant for slow depression component. This is common for all"
                               " synapses on this channel."
                               ),
                new ValueFinfo("tauF",
                               ValueFtype1< double >::global(),
                               GFCAST( &STPSynChan::getTauF),
                               RFCAST( &STPSynChan::setTauF),
                               "Time constant for facilitation component. This is common for all"
                               " synapses on this channel."
                               ),
                new ValueFinfo("deltaF",
                               ValueFtype1< double >::global(),
                               GFCAST( &STPSynChan::getDeltaF ),
                               RFCAST( &STPSynChan::setDeltaF ),
                               "Increment in facilitation due to each spike. This is common for all"
                               " synapses on this channel."
                               ),
                new ValueFinfo("d1",
                               ValueFtype1< double >::global(),
                               GFCAST( &STPSynChan::get_d1 ),
                               RFCAST( &STPSynChan::set_d1 ),
                               "Multiplicative increase in fast depression component for each incoming"
                               " spike. This is common for all synapses on this channel."
                               ),
                new ValueFinfo("d2",
                               ValueFtype1< double >::global(),
                               GFCAST( &STPSynChan::get_d2 ),
                               RFCAST( &STPSynChan::set_d2 ),
                               "Multiplicative increase in slow depression component for each incoming"
                               " spike. This is common for all synapses on this channel."
                               ),
                new LookupFinfo("D1",
                                LookupFtype< double, unsigned int >::global(),
                               GFCAST( &STPSynChan::getD1 ),
                               RFCAST( &dummyFunc ),
                               "First depression component"
                               ),
                new LookupFinfo("D2",
                                LookupFtype< double, unsigned int >::global(),
                               GFCAST( &STPSynChan::getD2 ),
                               RFCAST( &dummyFunc ),
                               "Second depression component."
                               ),
                new LookupFinfo("initF",
                                LookupFtype< double, unsigned int >::global(),
                                GFCAST( &STPSynChan::getInitF),
                                RFCAST( &STPSynChan::setInitF),
                                "Initial value for facilitation component. "
                                ),
                new LookupFinfo("F",
                                LookupFtype< double, unsigned int >::global(),
                                GFCAST( &STPSynChan::getF),
                                RFCAST( &dummyFunc),
                                "Value for facilitation component."
                                ),
                new LookupFinfo("initD1",
                                LookupFtype< double, unsigned int >::global(),
                                GFCAST( &STPSynChan::getInitD1 ),
                                RFCAST( &STPSynChan::setInitD1 ),
                                "Initial value for fast depression component."
                                ),
                new LookupFinfo("initD2",
                                LookupFtype< double, unsigned int >::global(),
                                GFCAST( &STPSynChan::getInitD2 ),
                                RFCAST( &STPSynChan::setInitD2 ),
                                "Initial value for slow depression component."
                                ),
                
            };

    static string doc[] =
            {
		"Name", "STPSynChan",
		"Author", "Subhasis Ray, 2011, NCBS",
		"Description", "STPSynChan: Synaptic channel incorporating short term plasticity."
                " This uses the formulation in the paper: Varela, J.A. et al. A"
                " Quantitative Description of Short-Term Plasticity at Excitatory"
                " Synapses in Layer 2/3 of Rat Primary Visual Cortex. The Journal of"
                " Neuroscience 17, 7926 -7940 (1997).\n"
                " But in stead of amplitude, this implementation modifies the"
                " transmitter release probability as:"
                " pr = pr0 * F * D1 * D2\n"
                " Whenever a release happens, the post-synaptic response is computed as"
                " usual SynChan objects, i.e., either as an alpha function or as a"
                " double exponential."
                " At each incoming spike:"
                " F <- F + deltaF"
                " D1 <- d1 * D1"
                " D2 <- d2 * D2.\n"
                " Otherwise,"
                " tauF * dF/dt = 1 - F\n"
                " tauD1 * d D1 / dt = 1 - D1\n"
                " tauD2 * d D2 / dt = 1 - D2.\n"
                " All the synapses converging on an STPSynChan share the values of "
                " deltaF, d1, d2, tauF, tauD1, tauD2 for the short term plasticity. But"
                " have independent F, D1 and D2 based on the history of incoming"
                " events."
            };
    static Cinfo STPSynChanCinfo(
            doc,
            sizeof( doc ) / sizeof( string ),		
            initSynChanCinfo(),
            STPSynChanFinfos,
            sizeof( STPSynChanFinfos )/sizeof(Finfo *),
            ValueFtype1< STPSynChan >::global());

    return &STPSynChanCinfo;
}

static const Cinfo* STPSynChanCinfo = initSTPSynChanCinfo();

static const Slot channelSlot =
	initSTPSynChanCinfo()->getSlot( "channel.channel" );
static const Slot origChannelSlot =
	initSTPSynChanCinfo()->getSlot( "origChannel" );
static const Slot gkSlot =
	initSTPSynChanCinfo()->getSlot( "GkSrc" );
static const Slot ikSlot =
	initSTPSynChanCinfo()->getSlot( "IkSrc" );

STPSynChan::STPSynChan(): d1_(1.0), d2_(1.0), deltaF_(0.0), tauD1_(1.0), tauD2_(1.0), tauF_(1.0), dt_tauF_(0.0), dt_tauD1_(0.0), dt_tauD2_(0.0)
{
    ;
}


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////


void STPSynChan::setInitPr( const Conn * conn, double val, const unsigned int& i )
{
    static_cast< STPSynChan* >(conn->data())->innerSetInitPr(conn->target(), val, i);
}

void STPSynChan::innerSetInitPr( Eref e, double val, const unsigned int& i )
{
    unsigned int size = updateNumSynapse(e);
    if ( i < size ){
        initPr_[i] = val;
    } else {
        cout << "Error: STPSynChan::setInitPr : Index " << i << 
                " out of range\n";
    }
}

double STPSynChan::getInitPr( Eref e, const unsigned int& i)
{
    return static_cast< STPSynChan * >(e.data())->innerGetInitPr(e, i);
}

double STPSynChan::innerGetInitPr( Eref e, const unsigned int& i )
{
    unsigned int size = updateNumSynapse(e);
    if ( i < size ){
        return initPr_[i];
    }
    cout << "Error: STPSynChan::getInitPr : Index " << i << 
            " out of range\n";
    return 0.0;
}

double STPSynChan::getPr( Eref e, const unsigned int& i)
{
    return static_cast< STPSynChan * >(e.data())->innerGetPr(e, i);
}

double STPSynChan::innerGetPr( Eref e, const unsigned int& i )
{
    unsigned int size = updateNumSynapse(e);
    if ( i < size ){
        return amp_[i] * (F_[i] + deltaF_) * d1_ * D1_[i] * d2_ * D2_[i];
    }
    cout << "Error: STPSynChan::getPr : Index " << i << 
            " out of range\n";
    return 0.0;
}

void STPSynChan::setInitF( const Conn * conn, double val, const unsigned int& i )
{
    static_cast< STPSynChan* >(conn->data())->innerSetInitF(conn->target(), val, i);
}

void STPSynChan::innerSetInitF( Eref e, double val, const unsigned int& i )
{
    unsigned int size = updateNumSynapse(e);
    if ( i < size ){
        initF_[i] = val;
    } else {
        cout << "Error: STPSynChan::setInitF : Index " << i << 
                " out of range\n";
    }
}

double STPSynChan::getInitF( Eref e, const unsigned int& i)
{
    return static_cast< STPSynChan * >(e.data())->innerGetInitF(e, i);
}

double STPSynChan::innerGetInitF( Eref e, const unsigned int& i )
{
    unsigned int size = updateNumSynapse(e);
    if ( i < size ){
        return initF_[i];
    }
    cout << "Error: STPSynChan::getInitF : Index " << i << 
            " out of range\n";
    return 0.0;
}


void STPSynChan::setInitD1( const Conn * conn, double val, const unsigned int& i )
{
    static_cast< STPSynChan* >(conn->data())->innerSetInitD1(conn->target(), val, i);
}

void STPSynChan::innerSetInitD1( Eref e, double val, const unsigned int& i )
{
    unsigned int size = updateNumSynapse(e);
    if ( i < size ){
        initD1_[i] = val;
    } else {
        cout << "Error: STPSynChan::setInitD1 : Index " << i << 
                " out of range\n";
    }
}

double STPSynChan::getInitD1( Eref e, const unsigned int& i)
{
    return static_cast< STPSynChan * >(e.data())->innerGetInitD1(e, i);
}

double STPSynChan::innerGetInitD1( Eref e, const unsigned int& i )
{
    unsigned int size = updateNumSynapse(e);
    if ( i < size ){
        return initD1_[i];
    }
    cout << "Error: STPSynChan::getInitD1 : Index " << i << 
            " out of range\n";
    return 0.0;
}

void STPSynChan::setInitD2( const Conn * conn, double val, const unsigned int& i )
{
    static_cast< STPSynChan* >(conn->data())->innerSetInitD2(conn->target(), val, i);
}

void STPSynChan::innerSetInitD2( Eref e, double val, const unsigned int& i )
{
    unsigned int size = updateNumSynapse(e);
    if ( i < size ){
        initD2_[i] = val;
    } else {
        cout << "Error: STPSynChan::setInitD2 : Index " << i << 
                " out of range\n";
    }
}

double STPSynChan::getInitD2( Eref e, const unsigned int& i)
{
    return static_cast< STPSynChan * >(e.data())->innerGetInitD2(e, i);
}

double STPSynChan::innerGetInitD2( Eref e, const unsigned int& i )
{
    unsigned int size = updateNumSynapse(e);
    if ( i < size ){
        return initD2_[i];
    }
    cout << "Error: STPSynChan::getInitD2 : Index " << i << 
            " out of range\n";
    return 0.0;
}

void STPSynChan::set_d1( const Conn * conn, double val)
{
    static_cast< STPSynChan* >(conn->data())->d1_ = val;
}


double STPSynChan::get_d1( Eref e)
{
    return static_cast< STPSynChan * >(e.data())->d1_;
}

void STPSynChan::set_d2( const Conn * conn, double val)
{
    static_cast< STPSynChan* >(conn->data())->d2_ = val;
}


double STPSynChan::get_d2( Eref e)
{
    return static_cast< STPSynChan * >(e.data())->d2_;
}

void STPSynChan::setTauF( const Conn * conn, double val)
{
    static_cast< STPSynChan* >(conn->data())->tauF_ = val;
}


double STPSynChan::getTauF( Eref e)
{
    return static_cast< STPSynChan * >(e.data())->tauF_;
}

void STPSynChan::setTauD1( const Conn * conn, double val)
{
    static_cast< STPSynChan* >(conn->data())->tauD1_ = val;
}


double STPSynChan::getTauD1( Eref e)
{
    return static_cast< STPSynChan * >(e.data())->tauD1_;
}
void STPSynChan::setTauD2( const Conn * conn, double val)
{
    static_cast< STPSynChan* >(conn->data())->tauD2_ = val;
}


double STPSynChan::getTauD2( Eref e)
{
    return static_cast< STPSynChan * >(e.data())->tauD2_;
}
void STPSynChan::setDeltaF( const Conn * conn, double val)
{
    static_cast< STPSynChan* >(conn->data())->deltaF_ = val;
}


double STPSynChan::getDeltaF( Eref e)
{
    return static_cast< STPSynChan * >(e.data())->deltaF_;
}

double STPSynChan::getD1( Eref e, const unsigned int& index)
{
    return static_cast< STPSynChan * >(e.data())->D1_[index];
}

double STPSynChan::getD2( Eref e, const unsigned int& index)
{
    return static_cast< STPSynChan * >(e.data())->D2_[index];
}

double STPSynChan::getF( Eref e, const unsigned int& index)
{
    return static_cast< STPSynChan * >(e.data())->F_[index];
}


unsigned  int STPSynChan::updateNumSynapse( Eref e)
{
    static const Finfo * synFinfo = initSTPSynChanCinfo()->findFinfo( "synapse" );
    unsigned int size = e.e->numTargets( synFinfo->msg(), e.i );
    if ( size != synapses_.size() ){
        synapses_.resize(size);
        initPr_.resize(size, 1.0);
        initF_.resize(size, 1.0);
        initD1_.resize(size, 1.0);
        initD2_.resize(size, 1.0);
        F_.resize(size, 1.0);
        D1_.resize(size, 1.0);
        D2_.resize(size, 1.0);
        amp_.resize(size, 1.0);
    }
    return size;
}


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void STPSynChan::innerProcessFunc( Eref e, ProcInfo info )
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
    for (unsigned int ii = 0; ii < synapses_.size(); ++ii){
        F_[ii] += (1 - F_[ii]) * (1 - 0.5 * dt_tauF_) * dt_tauF_;
        D1_[ii] += (1 - D1_[ii]) * (1 - 0.5 * dt_tauD2_) * dt_tauD1_;
        D2_[ii] += (1 - D2_[ii]) * (1 - 0.5 * dt_tauD2_) * dt_tauD2_;
    }
    send2< double, double >( e, channelSlot, Gk_, Ek_ );
    send2< double, double >( e, origChannelSlot, Gk_, Ek_ );
    send1< double >( e, ikSlot, Ik_ );
    send1< double >( e, gkSlot, Gk_ );    
}

/*
 * Note that this causes issues if we have variable dt.
 */
void STPSynChan::innerReinitFunc( Eref e, ProcInfo info )
{
    SynChan::innerReinitFunc(e, info);
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
    

void STPSynChan::innerSynapseFunc( const Conn* c, double time )
{
    unsigned int index = c->targetIndex();
    if ( index > synapses_.size() ){
        cout << "STPSynChan::innerSynapseFunc:: Index: " << index << ", size: " << synapses_.size() << endl;
        return;
    }
    F_[index] += deltaF_;
    D1_[index] *= d1_;
    D2_[index] *= d2_;    
    if ( mtrand() < amp_[index] * F_[index] * D1_[index] * D2_[index] )
    {
        pendingEvents_.push(synapses_[index].event( time ));
    }
}

#endif
