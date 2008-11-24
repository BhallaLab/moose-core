/*******************************************************************
 * File:            BinSynchan.cpp
 * Description:      Implements a subtype of SynChan where the
 *                      relase is probabilistic with binomial
 *                      distribution of release count at each
 *                      individual synapse.
 * Author:          Subhasis Ray
 * E-mail:          ray.subhasis@gmail.com
 * Created:         2007-11-26 11:44:01
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

#ifndef _BINSYNCHAN_CPP
#define _BINSYNCHAN_CPP
#include <math.h>
#include "moose.h"
#include <queue>

#include "BinSynchan.h"

static const double SynE = 2.7182818284590452354;

const Cinfo* initBinSynchanCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &BinSynchan::processFunc ) ),
	    new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &BinSynchan::reinitFunc ) ),
	};
	static Finfo* process =	new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ),
			"This is a shared message to receive Process message from the scheduler." );
		 
	static Finfo* channelShared[] =
	{
		new SrcFinfo( "channel", Ftype2< double, double >::global() ),
		new DestFinfo( "Vm", Ftype1< double >::global(), 
				RFCAST( &BinSynchan::channelFunc ) ),
	};

///////////////////////////////////////////////////////
// Field definitions
///////////////////////////////////////////////////////

	static Finfo* binSynchanFinfos[] =
	{
		new ValueFinfo( "Gbar", ValueFtype1< double >::global(),
			GFCAST( &BinSynchan::getGbar ), 
			RFCAST( &BinSynchan::setGbar )
		),
		new ValueFinfo( "Ek", ValueFtype1< double >::global(),
			GFCAST( &BinSynchan::getEk ), 
			RFCAST( &BinSynchan::setEk )
		),
		new ValueFinfo( "tau1", ValueFtype1< double >::global(),
			GFCAST( &BinSynchan::getTau1 ), 
			RFCAST( &BinSynchan::setTau1 )
		),
		new ValueFinfo( "tau2", ValueFtype1< double >::global(),
			GFCAST( &BinSynchan::getTau2 ), 
			RFCAST( &BinSynchan::setTau2 )
		),
		new ValueFinfo( "normalizeWeights", 
			ValueFtype1< bool >::global(),
			GFCAST( &BinSynchan::getNormalizeWeights ), 
			RFCAST( &BinSynchan::setNormalizeWeights )
		),
		new ValueFinfo( "Gk", ValueFtype1< double >::global(),
			GFCAST( &BinSynchan::getGk ), 
			RFCAST( &BinSynchan::setGk )
		),
		new ValueFinfo( "Ik", ValueFtype1< double >::global(),
			GFCAST( &BinSynchan::getIk ), 
			RFCAST( &BinSynchan::setIk )
		),

		new ValueFinfo( "numSynapses",
			ValueFtype1< unsigned int >::global(),
			GFCAST( &BinSynchan::getNumSynapses ), 
			&dummyFunc // Prohibit reassignment of this index.
		),

		new LookupFinfo( "weight",
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &BinSynchan::getWeight ),
			RFCAST( &BinSynchan::setWeight )
		),

		new LookupFinfo( "delay",
			LookupFtype< double, unsigned int >::global(),
			GFCAST( &BinSynchan::getDelay ),
			RFCAST( &BinSynchan::setDelay )
		),
                new LookupFinfo( "poolSize",
			LookupFtype< int, unsigned int >::global(),
			GFCAST( &BinSynchan::getPoolSize ),
			RFCAST( &BinSynchan::setPoolSize )
		),
                new LookupFinfo( "releaseP",
                                 LookupFtype< double, unsigned int>::global(),
                                 GFCAST( &BinSynchan::getReleaseP ),
                                 RFCAST( &BinSynchan::setReleaseP )
                    ),
                new LookupFinfo( "releaseCount",
                                 LookupFtype< double, unsigned int>::global(),
                                 GFCAST( &BinSynchan::getReleaseCount ),
                                 &dummyFunc
                    ),
                
///////////////////////////////////////////////////////
// Shared message definitions
///////////////////////////////////////////////////////
		process,
		new SharedFinfo( "channel", channelShared,
			sizeof( channelShared ) / sizeof( Finfo* ),
			"This is a shared message to couple channel to compartment. "
			"The first entry is a MsgSrc to send Gk and Ek to the compartment. "
			"The second entry is a MsgDest for Vm from the compartment. " ),

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
				RFCAST( &BinSynchan::synapseFunc ),
				"Arrival of a spike. Arg is time of sending of spike." ),

		new DestFinfo( "activation", Ftype1< double >::global(),
				RFCAST( &BinSynchan::activationFunc ), 
				"Sometimes we want to continuously activate the channel" ),

		new DestFinfo( "modulator", Ftype1< double >::global(),
				RFCAST( &BinSynchan::modulatorFunc ),
				"Modulate channel response" ),
	};

	// BinSynchan is scheduled after the compartment calculations.
	static SchedInfo schedInfo[] = { { process, 0, 1 } };

	static string doc[] =
	{
		"Name", "BinSynchan",
		"Author", "Upinder S. Bhalla, 2007, NCBS",
		"Description", "BinSynchan: Synaptic channel incorporating weight and delay. Does not "
				"handle activity-dependent modification, see HebbBinSynchan for that. "
				"Very similiar to the old synchan from GENESIS.", 
	};	
	static Cinfo binSynchanCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),
		initNeutralCinfo(),
		binSynchanFinfos,
		sizeof( binSynchanFinfos )/sizeof(Finfo *),
		ValueFtype1< BinSynchan >::global(),
		schedInfo, 1
	);

	return &binSynchanCinfo;
}

static const Cinfo* binSynchanCinfo = initBinSynchanCinfo();

static const Slot channelSlot =
	initBinSynchanCinfo()->getSlot( "channel" );
static const Slot origChannelSlot =
	initBinSynchanCinfo()->getSlot( "origChannel" );
static const Slot ikSlot =
	initBinSynchanCinfo()->getSlot( "IkSrc" );
static const Slot synapseSlot =
	initBinSynchanCinfo()->getSlot( "synapse" );


///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

void BinSynchan::setGbar( const Conn* c, double Gbar )
{
	static_cast< BinSynchan* >( c->data() )->Gbar_ = Gbar;
}
double BinSynchan::getGbar( Eref e )
{
	return static_cast< BinSynchan* >( e.data() )->Gbar_;
}

void BinSynchan::setEk( const Conn* c, double Ek )
{
	static_cast< BinSynchan* >( c->data() )->Ek_ = Ek;
}
double BinSynchan::getEk( Eref e )
{
	return static_cast< BinSynchan* >( e.data() )->Ek_;
}

void BinSynchan::setTau1( const Conn* c, double tau1 )
{
	static_cast< BinSynchan* >( c->data() )->tau1_ = tau1;
}
double BinSynchan::getTau1( Eref e )
{
	return static_cast< BinSynchan* >( e.data() )->tau1_;
}

void BinSynchan::setTau2( const Conn* c, double tau2 )
{
	static_cast< BinSynchan* >( c->data() )->tau2_ = tau2;
}
double BinSynchan::getTau2( Eref e )
{
	return static_cast< BinSynchan* >( e.data() )->tau2_;
}

void BinSynchan::setNormalizeWeights( const Conn* c, bool value )
{
	static_cast< BinSynchan* >( c->data() )->normalizeWeights_ = value;
}
bool BinSynchan::getNormalizeWeights( Eref e )
{
	return static_cast< BinSynchan* >( e.data() )->normalizeWeights_;
}

void BinSynchan::setGk( const Conn* c, double Gk )
{
	static_cast< BinSynchan* >( c->data() )->Gk_ = Gk;
}
double BinSynchan::getGk( Eref e )
{
	return static_cast< BinSynchan* >( e.data() )->Gk_;
}

void BinSynchan::setIk( const Conn* c, double Ik )
{
	static_cast< BinSynchan* >( c->data() )->Ik_ = Ik;
}
double BinSynchan::getIk( Eref e )
{
	return static_cast< BinSynchan* >( e.data() )->Ik_;
}

int BinSynchan::getNumSynapses( Eref e )
{
	return static_cast< BinSynchan* >( e.data() )->synapses_.size();
}

void BinSynchan::setWeight(
				const Conn* c, double val, const unsigned int& i )
{
	vector< BinSynInfo >& synapses = 
			static_cast< BinSynchan* >( c->data() )->synapses_;
	if ( i < synapses.size() )
		synapses[i].weight = val;
	else 
		cout << "Error: BinSynchan::setWeight: Index " << i << 
			" out of range\n";
}
double BinSynchan::getWeight( Eref e, const unsigned int& i )
{
	vector< BinSynInfo >& synapses = 
			static_cast< BinSynchan* >( e.data() )->synapses_;
	if ( i < synapses.size() )
		return synapses[i].weight;
	cout << "Error: BinSynchan::getWeight: Index " << i << 
			" out of range\n";
	return 0.0;
}

void BinSynchan::setDelay(
				const Conn* c, double val, const unsigned int& i )
{
	vector< BinSynInfo >& synapses = 
			static_cast< BinSynchan* >( c->data() )->synapses_;
	if ( i < synapses.size() )
		synapses[i].delay = val;
	else 
		cout << "Error: BinSynchan::setDelay: Index " << i << 
			" out of range\n";
}

double BinSynchan::getDelay( Eref e, const unsigned int& i )
{
	vector< BinSynInfo >& synapses = 
			static_cast< BinSynchan* >( e.data() )->synapses_;
	if ( i < synapses.size() )
		return synapses[i].delay;
	cout << "Error: BinSynchan::getDelay: Index " << i << 
			" out of range\n";
	return 0.0;
}

void BinSynchan::setPoolSize(
				const Conn* c, int val, const unsigned int& i )
{
	vector< BinSynInfo >& synapses = 
			static_cast< BinSynchan* >( c->data() )->synapses_;
	if ( i < synapses.size() )
            synapses[i].setPoolSize( val );
	else 
		cout << "Error: BinSynchan::setPoolSize : Index " << i << 
			" out of range\n";
}

int BinSynchan::getPoolSize( Eref e, const unsigned int& i )
{
	vector< BinSynInfo >& synapses = 
			static_cast< BinSynchan* >( e.data() )->synapses_;
	if ( i < synapses.size() )
            return synapses[i].getPoolSize();
	cout << "Error: BinSynchan::getPoolSize : Index " << i << 
			" out of range\n";
	return 0;
}

void BinSynchan::setReleaseP(
				const Conn* c, double val, const unsigned int& i )
{
	vector< BinSynInfo >& synapses = 
			static_cast< BinSynchan* >( c->data() )->synapses_;
	if ( i < synapses.size() )
            synapses[i].setReleaseP( val);
	else 
		cout << "Error: BinSynchan::setReleaseP : Index " << i << 
			" out of range\n";
}

double BinSynchan::getReleaseP( Eref e, const unsigned int& i )
{
	vector< BinSynInfo >& synapses = 
			static_cast< BinSynchan* >( e.data() )->synapses_;
	if ( i < synapses.size() )
            return synapses[i].getReleaseP();
	cout << "Error: BinSynchan::getReleaseP : Index " << i << 
			" out of range\n";
	return 0.0;
}

double BinSynchan::getReleaseCount( Eref e, const unsigned int& i )
{
	vector< BinSynInfo >& synapses = 
			static_cast< BinSynchan* >( e.data() )->synapses_;
	if ( i < synapses.size() )
            return synapses[i].getReleaseCount();
	cout << "Error: BinSynchan::getReleaseCount : Index " << i << 
			" out of range\n";
	return 0.0;
}

unsigned int BinSynchan::updateNumSynapse( Eref e )
{
	static const Finfo* synFinfo = initBinSynchanCinfo()->findFinfo( "synapse" );

	unsigned int n = e.e->numTargets( synFinfo->msg() );
	if ( n >= synapses_.size() )
			synapses_.resize( n );
	return synapses_.size();
}


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void BinSynchan::innerProcessFunc( Eref e, ProcInfo info )
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
void BinSynchan::processFunc( const Conn* c, ProcInfo p )
{
	static_cast< BinSynchan* >( c->data() )->
		innerProcessFunc( c->target(), p );
}

/*
 * Note that this causes issues if we have variable dt.
 */
void BinSynchan::innerReinitFunc( Eref e, ProcInfo info )
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
void BinSynchan::reinitFunc( const Conn* c, ProcInfo p )
{
	static_cast< BinSynchan* >( c->data() )->
		innerReinitFunc( c->target(), p );
}

void BinSynchan::innerSynapseFunc( const Conn* c, double time )
{
	unsigned int index = c->targetIndex();
	// Actually we should simply ignore any message where the
	// index is bigger than synapses_.size(), because the syn
	// strength will not yet have been set.
	assert( index < synapses_.size() );
	// The event function generates a new BinSynInfo object with the
	// time argument added to the delay_ field.
	// This goes into a priority_queue sorted by delay_.
	pendingEvents_.push( SynInfo(synapses_[index].weight, synapses_[index].delay + time ) );
}
void BinSynchan::synapseFunc( const Conn* c, double time )
{
	static_cast< BinSynchan* >( c->data() )->innerSynapseFunc( c, time );
}

void BinSynchan::channelFunc( const Conn* c, double Vm )
{
	static_cast< BinSynchan* >( c->data() )->Vm_ = Vm;
}

void BinSynchan::activationFunc( const Conn* c, double val )
{
	static_cast< BinSynchan* >( c->data() )->activation_ += val;
}

void BinSynchan::modulatorFunc( const Conn* c, double val )
{
	static_cast< BinSynchan* >( c->data() )->modulation_ *= val;
}


#endif
