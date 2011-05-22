/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <queue>
#include "header.h"
#include "Synapse.h"
#include "SynHandler.h"
#include "IntFire.h"

static SrcFinfo1< double > spike( 
		"spike", 
		"Sends out spike events"
	);

const Cinfo* IntFire::initCinfo()
{
		//////////////////////////////////////////////////////////////
		// Field Definitions
		//////////////////////////////////////////////////////////////
		static ValueFinfo< IntFire, double > Vm(
			"Vm",
			"Membrane potential",
			&IntFire::setVm,
			&IntFire::getVm
		);

		static ValueFinfo< IntFire, double > tau(
			"tau",
			"charging time-course",
			&IntFire::setTau,
			&IntFire::getTau
		);

		static ValueFinfo< IntFire, double > thresh(
			"thresh",
			"firing threshold",
			&IntFire::setThresh,
			&IntFire::getThresh
		);

		static ValueFinfo< IntFire, double > refractoryPeriod(
			"refractoryPeriod",
			"Minimum time between successive spikes",
			&IntFire::setRefractoryPeriod,
			&IntFire::getRefractoryPeriod
		);

		/*
		static ValueFinfo< IntFire, unsigned int > numSynapses(
			"numSynapses",
			"Number of synapses on IntFire",
			&IntFire::setNumSynapses,
			&IntFire::getNumSynapses
		);
		*/
		//////////////////////////////////////////////////////////////
		// MsgDest Definitions
		//////////////////////////////////////////////////////////////
		static DestFinfo process( "process",
			"Handles process call",
			new ProcOpFunc< IntFire >( &IntFire::process ) );
		static DestFinfo reinit( "reinit",
			"Handles reinit call",
			new ProcOpFunc< IntFire >( &IntFire::reinit ) );

		/*
		//////////////////////////////////////////////////////////////
		// FieldElementFinfo definition for Synapses
		//////////////////////////////////////////////////////////////
		static FieldElementFinfo< IntFire, Synapse > synFinfo( "synapse",
			"Sets up field Elements for synapse",
			Synapse::initCinfo(),
			&IntFire::getSynapse,
			&IntFire::setNumSynapses,
			&IntFire::getNumSynapses
		);
		*/
		//////////////////////////////////////////////////////////////
		// SharedFinfo Definitions
		//////////////////////////////////////////////////////////////
		static Finfo* procShared[] = {
			&process, &reinit
		};
		static SharedFinfo proc( "proc",
			"Shared message for process and reinit",
			procShared, sizeof( procShared ) / sizeof( const Finfo* )
		);

	static Finfo* intFireFinfos[] = {
		&Vm,	// Value
		&tau,	// Value
		&thresh,				// Value
		&refractoryPeriod,		// Value
		// &numSynapses,			// Value
		&proc,					// SharedFinfo
		&spike, 		// MsgSrc
		// &synFinfo		// FieldElementFinfo for synapses.
	};

	static Cinfo intFireCinfo (
		"IntFire",
		SynHandler::initCinfo(),
		intFireFinfos,
		sizeof( intFireFinfos ) / sizeof ( Finfo* ),
		new Dinfo< IntFire >()
	);

	return &intFireCinfo;
}

static const Cinfo* intFireCinfo = IntFire::initCinfo();

IntFire::IntFire()
	: Vm_( 0.0 ), thresh_( 0.0 ), tau_( 1.0 ), refractoryPeriod_( 0.1 ), lastSpike_( -0.1 )
{
	;
}

IntFire::IntFire( double thresh, double tau )
	: Vm_( 0.0 ), thresh_( thresh ), tau_( tau ), refractoryPeriod_( 0.1 ), lastSpike_( -0.1 )
{
	;
}

void IntFire::process( const Eref &e, ProcPtr p )
{
	/*
	if ( e.index().data() % 100 == 0 ) {
		cout << Shell::myNode() << "." << p->threadIndexInGroup << ":IntFire[" << e.index().data() << "]::process dt = " << p->dt << ", " << " t= " << p->currTime << ", #Pending= " << pendingEvents_.size() << ", Vm= " << Vm_ << endl << flush;
	}
	*/
	/*
	if ( e.index().data() == 1023 && pendingEvents_.size() > 0 && p->currTime > 0.9 ) {
		cout << "pending size on " << e.index() << " = " << pendingEvents_.size() << endl;
		*/
		/*
		while ( !pendingEvents_.empty() ) {
			double v = pendingEvents_.top().getWeight();
			double d = pendingEvents_.top().getDelay();
			cout << "(" << v << "," << d << ")	";
			pendingEvents_.pop();
		}
	}
	*/
	while ( !pendingEvents_.empty() &&
		pendingEvents_.top().getDelay() <= p->currTime ) {
			Vm_ += pendingEvents_.top().getWeight();
			pendingEvents_.pop();
	}
	if (  ( p->currTime - lastSpike_ ) < refractoryPeriod_ )
		Vm_ = 0.0;

	if ( Vm_ > thresh_ ) {
		spike.send( e, p, p->currTime );
		// e.sendSpike( spikeSlot, p->currTime );
		if ( e.index().data() % 100 == 0 ) {
			// cout << "IntFire[" << e.index().data() << "]::process, zeroing Vm= " << Vm_ << ", Ptr = " << this << endl;
		}
		Vm_ = -1.0e-7;
	} else {
		Vm_ *= ( 1.0 - p->dt / tau_ );
	}


/* This is what we would do for a conductance  channel.
	X_ = activation * xconst1_ + X_ * xconst2_;
	Y_ = X_ * yconst1_ + Y_ * yconst2_;
	*/
	 
/*
	unsigned int synSize = sizeof( SynInfo );
	for( char* i = e.processQ.begin(); i != e.processQ.end(); i += synSize )
	{
		SynInfo* si = static_cast< SynInfo* >( i );
		insertQ( si );
	}
	
	SynInfo* si = processQ.top();
	double current = 0.0;
	while ( si->time < p->time && si != processQ.end() ) {
		current += si->weight;
	}

	v_ += current * Gm_ + Em_ - tau_ * v_;
	if ( v_ > vThresh ) {
		v_ = Em_;
		sendWithId< double >( e, spikeSlot, p->t );
	}
*/
}


/**
 * Inserts an event into the pendingEvents queue for spikes.
 * Note that this function lives on the Element managing the Synapses,
 * and gets redirected to the IntFire.
 * This is called by UpFunc1< double >
 */
void IntFire::innerAddSpike( DataId index, const double time )
{
	assert( index.field() < getNumSynapses() );
	Synapse s( *getSynapse( index.field() ), time );
	// cout << index << "	";
	pendingEvents_.push( s );
}

void IntFire::reinit( const Eref& e, ProcPtr p )
{
	// pendingEvents_.resize( 0 );
	while( !pendingEvents_.empty() )
		pendingEvents_.pop();
	Vm_ = 0.0;
}

void IntFire::setVm( const double v )
{
	Vm_ = v;
}

void IntFire::setTau( const double v )
{
	tau_ = v;
}

void IntFire::setThresh( const double v )
{
	thresh_ = v;
}

void IntFire::setRefractoryPeriod( const double v )
{
	refractoryPeriod_ = v;
	lastSpike_ = -v;
}

double IntFire::getVm() const
{
	return Vm_;
}

double IntFire::getTau() const
{
	return tau_;
}

double IntFire::getThresh() const
{
	return thresh_;
}

double IntFire::getRefractoryPeriod() const
{
	return refractoryPeriod_;
}
