/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <queue>
#include "header.h"
// #include "SynInfo.h"
#include "SpikeRingBuffer.h"
#include "Synapse.h"
#include "SynHandler.h"
#include "ChanBase.h"
#include "SynChanBase.h"
#include "SynChan.h"

const double& SynE() {
	static const double SynE = exp(1.0);
	return SynE;
}

const Cinfo* SynChan::initCinfo()
{
	///////////////////////////////////////////////////////
	// Shared message definitions
	///////////////////////////////////////////////////////
	static DestFinfo process( "process", 
		"Handles process call",
		new ProcOpFunc< SynChan >( &SynChan::process ) );
	static DestFinfo reinit( "reinit", 
		"Handles reinit call",
		new ProcOpFunc< SynChan >( &SynChan::reinit ) );

	static Finfo* processShared[] =
	{
		&process, &reinit
	};

	static SharedFinfo proc( "proc", 
		"Shared message to receive Process message from scheduler",
		processShared, sizeof( processShared ) / sizeof( Finfo* ) );
		
///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////
	static ValueFinfo< SynChan, double > tau1( "tau1", 
		"Decay time constant for the synaptic conductance, tau1 >= tau2.",
        &SynChan::setTau1,
		&SynChan::getTau1
	);
	static ValueFinfo< SynChan, double > tau2( "tau2",
		"Rise time constant for the synaptic conductance, tau1 >= tau2.",
        &SynChan::setTau2,
		&SynChan::getTau2
	);
	static ValueFinfo< SynChan, bool > normalizeWeights( 
		"normalizeWeights", 
		"Flag. If true, the overall conductance is normalized by the "
		"number of individual synapses in this SynChan object.",
        &SynChan::setNormalizeWeights,
		&SynChan::getNormalizeWeights
	);

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
	static DestFinfo activation( "activation", 
		"Sometimes we want to continuously activate the channel",
		new OpFunc1< SynChan, double >( &SynChan::activation )
	);
	static DestFinfo modulator( "modulator",
		"Modulate channel response",
		new OpFunc1< SynChan, double >( &SynChan::modulator )
	);

	static Finfo* SynChanFinfos[] =
	{
		&proc,			// Shared
		&tau1,			// Value
		&tau2,			// Value
		&normalizeWeights,	// Value
		&activation,	// Dest
		&modulator,	// Dest
	};

	static string doc[] =
	{
		"Name", "SynChan",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "SynChan: Synaptic channel incorporating weight and delay. Does not "
				"handle activity-dependent modification, see HebbSynChan for that. "
				"Very similiar to the old synchan from GENESIS.", 
	};

        static Dinfo< SynChan > dinfo;

	static Cinfo SynChanCinfo(
		"SynChan",
		SynChanBase::initCinfo(),
		SynChanFinfos,
		sizeof( SynChanFinfos )/sizeof(Finfo *),
                &dinfo
	);

	return &SynChanCinfo;
}

static const Cinfo* synChanCinfo = SynChan::initCinfo();

SynChan::SynChan()
	: 
	tau1_( 1.0e-3 ), tau2_( 1.0e-3 ),
	normalizeWeights_( 0 ),
        xconst1_(0.0),
        yconst1_(1.0),
        xconst2_(1.0),
        yconst2_(0.0),
        norm_(1.0),
        activation_(0.0),
        modulation_(1.0),
        X_(0.0),
        Y_(0.0)    
{ ; }

SynChan::~SynChan()
{;}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
//
void SynChan::innerSetGbar( double Gbar )
{
	SynChanBase::innerSetGbar( Gbar );
	normalizeGbar();
}

void SynChan::setTau1( double tau1 )
{
	tau1_ = tau1;
}

double SynChan::getTau1() const
{
	return tau1_;
}

void SynChan::setTau2( double tau2 )
{
    tau2_ = tau2;
}

double SynChan::getTau2() const
{
    return tau2_;
}

void SynChan::setNormalizeWeights( bool value )
{
	normalizeWeights_ = value;
}

bool SynChan::getNormalizeWeights() const
{
	return normalizeWeights_;
}

void SynChan::normalizeGbar()
{
        if ( doubleEq( tau2_, 0.0 ) ) {
                // norm_ = 1.0;
                norm_ = SynChanBase::getGbar();
        } else {
                if ( doubleEq( tau1_, tau2_ ) ) {
                    norm_ = SynChanBase::getGbar() * SynE() / tau1_;
                } else {
                    double tpeak = tau1_ * tau2_ * log( tau1_ / tau2_ ) / 
                            ( tau1_ - tau2_ );
                    norm_ = SynChanBase::getGbar() * ( tau1_ - tau2_ ) / 
                            ( tau1_ * tau2_ * ( 
                            exp( -tpeak / tau1_ ) - exp( -tpeak / tau2_ )
                                                ));
                }
        }
	if ( normalizeWeights_ && getNumSynapses() > 0 )
		norm_ /= static_cast< double >( getNumSynapses() );
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void SynChan::process( const Eref& e, ProcPtr info )
{
	// while ( !pendingEvents_.empty() &&
		// pendingEvents_.top().getDelay() <= info->currTime ) {
        activation_ += popBuffer(info->currTime) / info->dt;
	// 	pendingEvents_.pop();
	// }
	X_ = modulation_ * activation_ * xconst1_ + X_ * xconst2_;
	Y_ = X_ * yconst1_ + Y_ * yconst2_;
	double Gk = Y_ * norm_;
	setGk( Gk );
	updateIk();
	activation_ = 0.0;
	modulation_ = 1.0;
	SynChanBase::process( e, info ); // Sends out messages for channel.
}

/*
 * Note that this causes issues if we have variable dt.
 */
void SynChan::reinit( const Eref& e, ProcPtr info )
{
	double dt = info->dt;
	activation_ = 0.0;
	modulation_ = 1.0;
	SynChanBase::setGk( 0.0 );
	SynChanBase::setIk( 0.0 );
	X_ = 0.0;
	Y_ = 0.0;
	xconst1_ = tau1_ * ( 1.0 - exp( -dt / tau1_ ) );
	xconst2_ = exp( -dt / tau1_ );

        if ( doubleEq( tau2_, 0.0 ) ) {
                yconst1_ = 1.0;
                yconst2_ = 0.0;
        } else {
                yconst1_ = tau2_ * ( 1.0 - exp( -dt / tau2_ ) );
                yconst2_ = exp( -dt / tau2_ );
        }
	normalizeGbar();
	/*
        if ( doubleEq( tau2_, 0.0 ) ) {
                yconst1_ = 1.0;
                yconst2_ = 0.0;
                norm_ = 1.0;
        } else {
                yconst1_ = tau2_ * ( 1.0 - exp( -dt / tau2_ ) );
                yconst2_ = exp( -dt / tau2_ );
                if ( tau1_ == tau2_ ) {
                    norm_ = SynChanBase::getGbar() * SynE() / tau1_;
                } else {
                    double tpeak = tau1_ * tau2_ * log( tau1_ / tau2_ ) / 
                            ( tau1_ - tau2_ );
                    norm_ = SynChanBase::getGbar() * ( tau1_ - tau2_ ) / 
                            ( tau1_ * tau2_ * ( 
                            exp( -tpeak / tau1_ ) - exp( -tpeak / tau2_ )
                                                ));
                }
        }
        
	// updateNumSynapse( e );
	if ( normalizeWeights_ && getNumSynapses() > 0 )
		norm_ /= static_cast< double >( getNumSynapses() );
	*/
	// while ( !pendingEvents_.empty() )
	// 	pendingEvents_.pop();
        SynChanBase::reinit(e, info);
}

void SynChan::activation( double val )
{
	activation_ += val;
}

void SynChan::modulator( double val )
{
	modulation_ *= val;
}

///////////////////////////////////////////////////
// Utility function
///////////////////////////////////////////////////
/*
* Turns out to be rather hard.
unsigned int SynChan::countNumSynapses( const Eref& er )
{
	static const DestFinfo* spikeFinfo = 
		dynamic_cast< const DestFinfo* >(
			Synapse::initCinfo()->findFinfo("addSpike") );
	assert( spikeFinfo != 0 );
	assert( reinterpret_cast< char* >(this) == er.data() );
	Id synId( er.id().value() + 1 );
}
*/

/*
 * Not needed anymore as the synapses get the spike directly from SpikeGens.
void SynChan::innerAddSpike( unsigned int synIndex, const double time )
{
	assert( synIndex < getNumSynapses() );
	Synapse s( *getSynapse( synIndex ));
        s.setDelay(time);
	// pendingEvents_.push( s );
}
*/
