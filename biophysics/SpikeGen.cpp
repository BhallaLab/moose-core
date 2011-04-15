/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <math.h>

#include "SpikeGen.h"

const Cinfo* initSpikeGenCinfo()
{
	static Finfo* processShared[] =
	{
		new DestFinfo( "process", Ftype1< ProcInfo >::global(),
				RFCAST( &SpikeGen::processFunc ) ),
		new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
				RFCAST( &SpikeGen::reinitFunc ) ),
	};
	static Finfo* process =	new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ),
			"This is a shared message to receive Process messages from the scheduler objects." );
	
	static Finfo* spikeGenFinfos[] = 
	{
		new ValueFinfo( "threshold", ValueFtype1< double >::global(),
			GFCAST( &SpikeGen::getThreshold ),
			RFCAST( &SpikeGen::setThreshold )
		),
		new ValueFinfo( "refractT", ValueFtype1< double >::global(),
			GFCAST( &SpikeGen::getRefractT ),
			RFCAST( &SpikeGen::setRefractT )
		),
		new ValueFinfo( "abs_refract", ValueFtype1< double >::global(),
			GFCAST( &SpikeGen::getRefractT ),
			RFCAST( &SpikeGen::setRefractT )
		),

		/**	"The amplitude field of the spikeGen is never used. "
		 	\todo: perhaps should deprecate this."	*/

		new ValueFinfo( "amplitude", ValueFtype1< double >::global(),
			GFCAST( &SpikeGen::getAmplitude ),
			RFCAST( &SpikeGen::setAmplitude ),
			"The amplitude field of the spikeGen is never used. "
		),
		new ValueFinfo( "state", ValueFtype1< double >::global(),
			GFCAST( &SpikeGen::getState ),
			RFCAST( &SpikeGen::setState )
		),
		new ValueFinfo( "edgeTriggered", ValueFtype1< int >::global(),
			GFCAST( &SpikeGen::getEdgeTriggered ),
                        RFCAST( &SpikeGen::setEdgeTriggered ),
                        "When edgeTriggered = 0, the SpikeGen will fire an event in each timestep "
                        "while incoming Vm is > threshold and at least abs_refract time has passed "
                        "since last event. This may be problematic if the incoming Vm remains above "
                        "threshold for longer than abs_refract. Setting edgeTriggered to 1 resolves this "
                        "as the SpikeGen generates an event only on the rising edge of the incoming Vm "
                        "and will remain idle unless the incoming Vm goes below threshold."
		),

	//////////////////////////////////////////////////////////////////
	// SharedFinfos
	//////////////////////////////////////////////////////////////////
		process,
		/*
		new SharedFinfo( "process", processShared, 
			sizeof( processShared ) / sizeof( Finfo* ) ),
			*/

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////
		new SrcFinfo( "event", Ftype1< double >::global(), 
			"Sends out a trigger for an event. The time is not sent - everyone knows the time." ),

	//////////////////////////////////////////////////////////////////
	// Dest Finfos.
	//////////////////////////////////////////////////////////////////
		new DestFinfo( "Vm", Ftype1< double >::global(),
			RFCAST( &SpikeGen::VmFunc ) ),
	};

	// We want the spikeGen to update after the compartments have done so
	static SchedInfo schedInfo[] = { { process, 0, 1 } };
	static string doc[] =
	{
		"Name", "SpikeGen",
		"Author", "Upi Bhalla",
		"Description", "SpikeGen object, for detecting threshold crossings. "
                "The threshold detection can work in multiple modes.\n "
                "If the refractT < 0.0, then it fires an event only at the rising edge of the "
	};
	static Cinfo spikeGenCinfo(
				doc,
				sizeof( doc ) / sizeof( string ),				
				initNeutralCinfo(),
				spikeGenFinfos,
				sizeof( spikeGenFinfos ) / sizeof( Finfo* ),
				ValueFtype1< SpikeGen >::global(),
				schedInfo, 1
	);

	return &spikeGenCinfo;
}

static const Cinfo* spikeGenCinfo = initSpikeGenCinfo();

static const Slot eventSlot =
	initSpikeGenCinfo()->getSlot( "event" );

//////////////////////////////////////////////////////////////////
// Here we put the SpikeGen class functions.
//////////////////////////////////////////////////////////////////

// Value Field access function definitions.
void SpikeGen::setThreshold( const Conn* c, double threshold )
{
	static_cast< SpikeGen* >( c->data() )->threshold_ = threshold;
}
double SpikeGen::getThreshold( Eref e )
{
	return static_cast< SpikeGen* >( e.data() )->threshold_;
}

void SpikeGen::setRefractT( const Conn* c, double val )
{
	static_cast< SpikeGen* >( c->data() )->refractT_ = val;
}
double SpikeGen::getRefractT( Eref e )
{
	return static_cast< SpikeGen* >( e.data() )->refractT_;
}

void SpikeGen::setAmplitude( const Conn* c, double val )
{
	static_cast< SpikeGen* >( c->data() )->amplitude_ = val;
}
double SpikeGen::getAmplitude( Eref e )
{
	return static_cast< const SpikeGen* >( e.data() )->amplitude_;
}

void SpikeGen::setState( const Conn* c, double val )
{
	static_cast< SpikeGen* >( c->data() )->state_ = val;
}
double SpikeGen::getState( Eref e )
{
	return static_cast< SpikeGen* >( e.data() )->state_;
}

void SpikeGen::setEdgeTriggered( const Conn* c, int yes )
{
	static_cast< SpikeGen* >( c->data() )->edgeTriggered_ = yes;
}
int SpikeGen::getEdgeTriggered( Eref e )
{
	return static_cast< SpikeGen* >( e.data() )->edgeTriggered_;
}

//////////////////////////////////////////////////////////////////
// SpikeGen::Dest function definitions.
//////////////////////////////////////////////////////////////////

void SpikeGen::innerProcessFunc( const Conn* c, ProcInfo p )
{
	double t = p->currTime_;
	if ( V_ > threshold_ ) {
            if ((t + p->dt_/2.0) >= (lastEvent_ + refractT_)){
                if (!edgeTriggered_ || (edgeTriggered_ && !fired_)) {
                    send1< double >( c->target(), eventSlot, t );
                    lastEvent_ = t;
                    state_ = amplitude_;
                    fired_ = true;                    
                } else {
                    state_ = 0.0;                
                }
            }
	} else {
            state_ = 0.0;
            fired_ = false;
	}
}
void SpikeGen::processFunc( const Conn* c, ProcInfo p )
{
	static_cast< SpikeGen* >( c->data() )->innerProcessFunc( c, p );
}

// Set it so that first spike is allowed.
void SpikeGen::reinitFunc( const Conn* c, ProcInfo p )
{
	static_cast< SpikeGen* >( c->data() )->lastEvent_ = 
			- static_cast< SpikeGen* >( c->data() )->refractT_ ;
}

void SpikeGen::VmFunc( const Conn* c, double val )
{
	static_cast< SpikeGen* >( c->data() )->V_ = val;
}

/////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
#include "../element/Neutral.h"

void testSpikeGen()
{
	cout << "\nTesting SpikeGen" << flush;

	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(), 
		Id::scratchId() );
	Element* sg = Neutral::create( "SpikeGen", "c0", n->id(), 
		Id::scratchId() );
	ASSERT( sg != 0, "creating compartment" );
	ProcInfoBase p;
	SetConn c( sg, 0 );
	p.dt_ = 0.001;
	p.currTime_ = 0.0;
	SpikeGen::setThreshold( &c, 1.0 );
	SpikeGen::setAmplitude( &c, 1.0 );
	SpikeGen::setRefractT( &c, 0.005 );
	SpikeGen::reinitFunc( &c, &p );

	SpikeGen::VmFunc( &c, 0.5 );
	SpikeGen::processFunc( &c, &p );
	ASSERT( SpikeGen::getState( sg ) == 0.0, "SpikeGen" );
	p.currTime_ += p.dt_;

	SpikeGen::VmFunc( &c, 0.999 );
	SpikeGen::processFunc( &c, &p );
	ASSERT( SpikeGen::getState( sg ) == 0.0, "SpikeGen" );
	p.currTime_ += p.dt_;

	SpikeGen::VmFunc( &c, 1.01 ); // First spike
	SpikeGen::processFunc( &c, &p );
	ASSERT( SpikeGen::getState( sg ) == 1.0, "SpikeGen" );
	p.currTime_ += p.dt_;

	SpikeGen::VmFunc( &c, 0.999 );
	SpikeGen::processFunc( &c, &p );
	ASSERT( SpikeGen::getState( sg ) == 0.0, "SpikeGen" );
	p.currTime_ += p.dt_;

	SpikeGen::VmFunc( &c, 2.0 ); // Too soon, refractory.
	SpikeGen::processFunc( &c, &p );
	ASSERT( SpikeGen::getState( sg ) == 0.0, "SpikeGen" );

	p.currTime_ = 0.010;
	SpikeGen::VmFunc( &c, 2.0 ); // Now not refractory.
	SpikeGen::processFunc( &c, &p );
	ASSERT( SpikeGen::getState( sg ) == 1.0, "SpikeGen" );

	// Get rid of all the test objects
	set( n, "destroy" );
}
#endif
