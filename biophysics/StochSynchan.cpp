/*******************************************************************
 * File:            StochSynchan.cpp
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

#ifndef _STOCHSYNCHAN_CPP
#define _STOCHSYNCHAN_CPP
#include <math.h>
#include "moose.h"
#include <queue>
#include "randnum/randnum.h"

#include "StochSynchan.h"

static const double SynE = 2.7182818284590452354;

const Cinfo* initStochSynchanCinfo()
{
	static Finfo* processShared[] =
	{
            new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &StochSynchan::processFunc ) ),
	    new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &StochSynchan::reinitFunc ) ),
	};
	static Finfo* process =	new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ),
			"This is a shared message to receive Process message from the scheduler." );
	static Finfo* channelShared[] =
	{
		new SrcFinfo( "channel", Ftype2< double, double >::global() ),
		new DestFinfo( "Vm", Ftype1< double >::global(), 
				RFCAST( &StochSynchan::channelFunc ) ),
	};

///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////

	static Finfo* stochSynchanFinfos[] =
	{
		new ValueFinfo( "Gbar", ValueFtype1< double >::global(),
			GFCAST( &StochSynchan::getGbar ), 
			RFCAST( &StochSynchan::setGbar )
		),
		new ValueFinfo( "Ek", ValueFtype1< double >::global(),
			GFCAST( &StochSynchan::getEk ), 
			RFCAST( &StochSynchan::setEk )
		),
		new ValueFinfo( "tau1", ValueFtype1< double >::global(),
			GFCAST( &StochSynchan::getTau1 ), 
			RFCAST( &StochSynchan::setTau1 )
		),
		new ValueFinfo( "tau2", ValueFtype1< double >::global(),
			GFCAST( &StochSynchan::getTau2 ), 
			RFCAST( &StochSynchan::setTau2 )
		),
		new ValueFinfo( "normalizeWeights", 
			ValueFtype1< bool >::global(),
			GFCAST( &StochSynchan::getNormalizeWeights ), 
			RFCAST( &StochSynchan::setNormalizeWeights )
		),
		new ValueFinfo( "Gk", ValueFtype1< double >::global(),
			GFCAST( &StochSynchan::getGk ), 
			RFCAST( &StochSynchan::setGk )
		),
		new ValueFinfo( "Ik", ValueFtype1< double >::global(),
			GFCAST( &StochSynchan::getIk ), 
			RFCAST( &StochSynchan::setIk )
		),

		new ValueFinfo( "numSynapses",
			ValueFtype1< unsigned int >::global(),
			GFCAST( &StochSynchan::getNumSynapses ), 
			&dummyFunc // Prohibit reassignment of this index.
		),

		new LookupFinfo( "weight",
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &StochSynchan::getWeight ),
			RFCAST( &StochSynchan::setWeight )
		),

		new LookupFinfo( "delay",
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &StochSynchan::getDelay ),
			RFCAST( &StochSynchan::setDelay )
		),
                new LookupFinfo( "releaseP",
                                 LookupFtype< double, unsigned int>::global(),
                                 GFCAST( &StochSynchan::getReleaseP ),
                                 RFCAST( &StochSynchan::setReleaseP )
                    ),
                new LookupFinfo( "releaseCount",
                                 LookupFtype< double, unsigned int>::global(),
                                 GFCAST( &StochSynchan::getReleaseCount ),
                                 &dummyFunc
                    ),
                
///////////////////////////////////////////////////////
// Shared message definitions
///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "process", processShared,
			sizeof( processShared ) / sizeof( Finfo* ) ), 
		new SharedFinfo( "channel", channelShared,
			sizeof( channelShared ) / sizeof( Finfo* ),
			"This is a shared message to couple channel to compartment. "
			"The first entry is a MsgSrc to send Gk and Ek to the compartment "
			"The second entry is a MsgDest for Vm from the compartment." ),

///////////////////////////////////////////////////////
// MsgSrc definitions
///////////////////////////////////////////////////////
		new SrcFinfo( "IkSrc", Ftype1< double >::global() ),
		new SrcFinfo( "origChannel", Ftype2< double, double >::
			global() ),

///////////////////////////////////////////////////////
// MsgDest definitions
///////////////////////////////////////////////////////
		new DestFinfo( "synapse", Ftype1< double >::global(),
				RFCAST( &StochSynchan::synapseFunc ),
				"Arrival of a spike. Arg is time of sending of spike." ),

		new DestFinfo( "activation", Ftype1< double >::global(),
				RFCAST( &StochSynchan::activationFunc ),
				"Sometimes we want to continuously activate the channel"),

		new DestFinfo( "modulator", Ftype1< double >::global(),
				RFCAST( &StochSynchan::modulatorFunc ),
				"Modulate channel response" ),
	};

	// StochSynchan is scheduled after the compartment calculations.
	static SchedInfo schedInfo[] = { { process, 0, 1 } };
	
	static string doc[] =
	{
		"Name", "StochSynchan",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "StochSynchan: Synaptic channel incorporating weight and delay. Does not "
				"handle activity-dependent modification, see HebbStochSynchan for that. "
				"Very similiar to the old synchan from GENESIS.", 
	};
	static Cinfo stochSynchanCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initNeutralCinfo(),
		stochSynchanFinfos,
		sizeof( stochSynchanFinfos )/sizeof(Finfo *),
		ValueFtype1< StochSynchan >::global(),
		schedInfo, 1
	);

	return &stochSynchanCinfo;
}

static const Cinfo* stochSynchanCinfo = initStochSynchanCinfo();

static const Slot channelSlot =
	initStochSynchanCinfo()->getSlot( "channel" );
static const Slot origChannelSlot =
	initStochSynchanCinfo()->getSlot( "origChannel" );
static const Slot ikSlot =
	initStochSynchanCinfo()->getSlot( "IkSrc" );
static const Slot synapseSlot =
	initStochSynchanCinfo()->getSlot( "synapse" );


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void StochSynchan::setGbar( const Conn* c, double Gbar )
{
	static_cast< StochSynchan* >( c->data() )->Gbar_ = Gbar;
}
double StochSynchan::getGbar( Eref e )
{
	return static_cast< StochSynchan* >( e.data() )->Gbar_;
}

void StochSynchan::setEk( const Conn* c, double Ek )
{
	static_cast< StochSynchan* >( c->data() )->Ek_ = Ek;
}
double StochSynchan::getEk( Eref e )
{
	return static_cast< StochSynchan* >( e.data() )->Ek_;
}

void StochSynchan::setTau1( const Conn* c, double tau1 )
{
	static_cast< StochSynchan* >( c->data() )->tau1_ = tau1;
}
double StochSynchan::getTau1( Eref e )
{
	return static_cast< StochSynchan* >( e.data() )->tau1_;
}

void StochSynchan::setTau2( const Conn* c, double tau2 )
{
	static_cast< StochSynchan* >( c->data() )->tau2_ = tau2;
}
double StochSynchan::getTau2( Eref e )
{
	return static_cast< StochSynchan* >( e.data() )->tau2_;
}

void StochSynchan::setNormalizeWeights( const Conn* c, bool value )
{
	static_cast< StochSynchan* >( c->data() )->normalizeWeights_ = value;
}
bool StochSynchan::getNormalizeWeights( Eref e )
{
	return static_cast< StochSynchan* >( e.data() )->normalizeWeights_;
}

void StochSynchan::setGk( const Conn* c, double Gk )
{
	static_cast< StochSynchan* >( c->data() )->Gk_ = Gk;
}
double StochSynchan::getGk( Eref e )
{
	return static_cast< StochSynchan* >( e.data() )->Gk_;
}

void StochSynchan::setIk( const Conn* c, double Ik )
{
	static_cast< StochSynchan* >( c->data() )->Ik_ = Ik;
}
double StochSynchan::getIk( Eref e )
{
	return static_cast< StochSynchan* >( e.data() )->Ik_;
}

int StochSynchan::getNumSynapses( Eref e )
{
	return static_cast< StochSynchan* >( e.data() )->synapses_.size();
}

void StochSynchan::setWeight(
				const Conn* c, double val, const unsigned int& i )
{
	vector< StochSynInfo >& synapses = 
			static_cast< StochSynchan* >( c->data() )->synapses_;
	if ( i < synapses.size() )
		synapses[i].weight = val;
	else 
		cout << "Error: StochSynchan::setWeight: Index " << i << 
			" out of range\n";
}
double StochSynchan::getWeight( Eref e, const unsigned int& i )
{
	vector< StochSynInfo >& synapses = 
			static_cast< StochSynchan* >( e.data() )->synapses_;
	if ( i < synapses.size() )
		return synapses[i].weight;
	cout << "Error: StochSynchan::getWeight: Index " << i << 
			" out of range\n";
	return 0.0;
}

void StochSynchan::setDelay(
				const Conn* c, double val, const unsigned int& i )
{
	vector< StochSynInfo >& synapses = 
			static_cast< StochSynchan* >( c->data() )->synapses_;
	if ( i < synapses.size() )
		synapses[i].delay = val;
	else 
		cout << "Error: StochSynchan::setDelay: Index " << i << 
			" out of range\n";
}

double StochSynchan::getDelay( Eref e, const unsigned int& i )
{
	vector< StochSynInfo >& synapses = 
			static_cast< StochSynchan* >( e.data() )->synapses_;
	if ( i < synapses.size() )
		return synapses[i].delay;
	cout << "Error: StochSynchan::getDelay: Index " << i << 
			" out of range\n";
	return 0.0;
}


void StochSynchan::setReleaseP(
				const Conn* c, double val, const unsigned int& i )
{
	vector< StochSynInfo >& synapses = 
			static_cast< StochSynchan* >( c->data() )->synapses_;
	if ( i < synapses.size() )
            synapses[i].releaseP = val;
	else 
		cout << "Error: StochSynchan::setReleaseP : Index " << i << 
			" out of range\n";
}

double StochSynchan::getReleaseP( Eref e, const unsigned int& i )
{
	vector< StochSynInfo >& synapses = 
			static_cast< StochSynchan* >( e.data() )->synapses_;
	if ( i < synapses.size() )
            return synapses[i].releaseP;
	cout << "Error: StochSynchan::getReleaseP : Index " << i << 
			" out of range\n";
	return 0.0;
}

double StochSynchan::getReleaseCount( Eref e, const unsigned int& i )
{
	vector< StochSynInfo >& synapses = 
			static_cast< StochSynchan* >( e.data() )->synapses_;
	if ( i < synapses.size() )
            return (synapses[i].hasReleased? 1.0 : 0.0);
	cout << "Error: StochSynchan::getReleaseCount : Index " << i << 
			" out of range\n";
	return 0.0;
}

unsigned int StochSynchan::updateNumSynapse( Eref e )
{
	static const Finfo* synFinfo = initStochSynchanCinfo()->findFinfo( "synapse" );

	unsigned int n = e.e->numTargets( synFinfo->msg() );
	if ( n >= synapses_.size() )
			synapses_.resize( n );
	return synapses_.size();
}



///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void StochSynchan::innerProcessFunc( Eref e, ProcInfo info )
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
}

void StochSynchan::processFunc( const Conn* c, ProcInfo p )
{
	static_cast< StochSynchan* >( c->data() )->
		innerProcessFunc( c->target(), p );
}

/*
 * Note that this causes issues if we have variable dt.
 */
void StochSynchan::innerReinitFunc( Eref e, ProcInfo info )
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
			) );
	}
	updateNumSynapse( e );
	if ( normalizeWeights_ && synapses_.size() > 0 )
		norm_ /= static_cast< double >( synapses_.size() );
	while ( !pendingEvents_.empty() )
		pendingEvents_.pop();
}
void StochSynchan::reinitFunc( const Conn* c, ProcInfo p )
{
	static_cast< StochSynchan* >( c->data() )->
		innerReinitFunc( c->target(), p );
}

void StochSynchan::innerSynapseFunc( const Conn* c, double time )
{
	unsigned int index = c->targetIndex();
	// Actually we should simply ignore any message where the
	// index is bigger than synapses_.size(), because the syn
	// strength will not yet have been set.
	assert( index < synapses_.size() );
	// The event function generates a new SynInfo object with the
	// time argument added to the delay_ field.
	// This goes into a priority_queue sorted by delay_.
        if ( mtrand() < synapses_[index].releaseP )
        {
            pendingEvents_.push(synapses_[index].event( time ));
            synapses_[index].hasReleased = true;            
        }
        else
        {
            synapses_[index].hasReleased = false;
        }
        
}
void StochSynchan::synapseFunc( const Conn* c, double time )
{
	static_cast< StochSynchan* >( c->data() )->innerSynapseFunc( c, time );
}

void StochSynchan::channelFunc( const Conn* c, double Vm )
{
	static_cast< StochSynchan* >( c->data() )->Vm_ = Vm;
}

void StochSynchan::activationFunc( const Conn* c, double val )
{
	static_cast< StochSynchan* >( c->data() )->activation_ += val;
}

void StochSynchan::modulatorFunc( const Conn* c, double val )
{
	static_cast< StochSynchan* >( c->data() )->modulation_ *= val;
}
#endif
