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

#include "Compartment.h"
#include "SymCompartment.h"


/**
 * The initCompartmentCinfo() function sets up the Compartment class.
 * This function uses the common trick of having an internal
 * static value which is created the first time the function is called.
 * There are several static arrays set up here. The ones which
 * use SharedFinfos are for shared messages where multiple kinds
 * of information go along the same connection.
 */
const Cinfo* initSymCompartmentCinfo()
{
	/**
	 * This is a shared message to receive Init messages from
	 * the scheduler objects.
	 * Its job is to separate the compartmental calculations from
	 * the inter-compartment message passing.
	 * It uses the same init function as the regular compartment.
	 * Unlike the regular compartment, it is important at the reinit
	 * stage, because we need to tally up Ra values to calculate the
	 * coeffs.
	 *
	static Finfo* initShared[] =
	{
		new DestFinfo( "init", Ftype1< ProcInfo >::global(),
				RFCAST( &Compartment::initFunc ) ),
		new DestFinfo( "initReinit", Ftype1< ProcInfo >::global(),
				RFCAST( &SymCompartment::initReinit ) ),
	};
	static Finfo* init = new SharedFinfo( "init", initShared,
			sizeof( initShared ) / sizeof( Finfo* ) );
	 */

	static Finfo* raxialShared[] =
	{
		new DestFinfo( "Raxial", Ftype2< double, double >::global(),
			RFCAST( &SymCompartment::raxialFunc ),
			"Expects Ra and Vm from other compartment." ),
		new DestFinfo( "sumRaxial", Ftype1< double >::global(),
			RFCAST( &SymCompartment::sumRaxial ),
			"Expects Ra from other compartment." ),
		new DestFinfo( "sumRaxialRequest", Ftype0::global(),
			RFCAST( &SymCompartment::sumRaxialRequest ),
			"Handles a request to send back Ra to originating compartment." ),
		new SrcFinfo( "RaxialSrc", Ftype2< double, double >::global(),
			"Sends out Ra and Vm" ),
		new SrcFinfo( "sumRaxialSrc", Ftype1< double >::global(), 
			"Sends out Ra" ),
		new SrcFinfo( "sumRaxialRequestSrc", Ftype0::global(), 
			"Sends out request for Ra." ),
	};

	static Finfo* raxial2Shared[] =
	{
		
		new DestFinfo( "Raxial2", Ftype2< double, double >::global(),
			RFCAST( &SymCompartment::raxial2Func ),
			"Expects Ra and Vm from other compartment." ),
		new DestFinfo( "sumRaxial2", Ftype1< double >::global(),
			RFCAST( &SymCompartment::sumRaxial2 ),
			"Expects Ra from other compartment." ),
		new DestFinfo( "sumRaxial2Request", Ftype0::global(),
			RFCAST( &SymCompartment::sumRaxial2Request ),
			"Handles a request to send back Ra to originating compartment." ),
		new SrcFinfo( "Raxial2Src", Ftype2< double, double >::global(),
			"Sends out Ra and Vm"),
		new SrcFinfo( "sumRaxial2Src", Ftype1< double >::global(),
			"Sends out Ra" ),
		new SrcFinfo( "sumRaxial2RequestSrc", Ftype0::global(),
			"Sends out request for Ra." ),
	};

	static Finfo* symCompartmentFinfos[] = 
	{

	//////////////////////////////////////////////////////////////////
	// SharedFinfo definitions
	//////////////////////////////////////////////////////////////////
	    // The inherited process and init messages do not need to be
		// overridden.
		
		// Lots of aliases for raxial and raxial2.
		new SharedFinfo( "raxial1", raxialShared,
			sizeof( raxialShared ) / sizeof( Finfo* ),
			"This is a raxial shared message between symmetric compartments." ),
		new SharedFinfo( "CONNECTTAIL", raxialShared,
			sizeof( raxialShared ) / sizeof( Finfo* ),
			"This is a raxial shared message between symmetric compartments." ),
		new SharedFinfo( "raxial2", raxial2Shared,
			sizeof( raxial2Shared ) / sizeof( Finfo* ),
			"This is a raxial2 shared message between symmetric compartments." ),
		new SharedFinfo( "CONNECTHEAD", raxial2Shared,
			sizeof( raxial2Shared ) / sizeof( Finfo* ) ),
		new SharedFinfo( "CONNECTCROSS", raxial2Shared,
			sizeof( raxial2Shared ) / sizeof( Finfo* ) ),

	///////////////////////////////////////////////////////
	// MsgSrc definitions
	///////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////
	// DestFinfo definitions
	//////////////////////////////////////////////////////////////////
	};

	// static SchedInfo schedInfo[] = { { process, 0, 0 }, { init, 0, 1 } };
	
	static string doc[] =
	{
		"Name", "SymCompartment",
		"Author", "Upi Bhalla",
		"Description", "SymCompartment object, for branching neuron models.",
	};
	static Cinfo symCompartmentCinfo(
				doc,
				sizeof( doc ) / sizeof( string ),
				initCompartmentCinfo(),
				symCompartmentFinfos,
				sizeof( symCompartmentFinfos ) / sizeof( Finfo* ),
				ValueFtype1< SymCompartment >::global()
				// I wonder if these are inherited properly?
				// schedInfo, 2
	);

	return &symCompartmentCinfo;
}

static const Cinfo* symCompartmentCinfo = initSymCompartmentCinfo();

static const Slot raxialSlot =
	initSymCompartmentCinfo()->getSlot( "raxial1.RaxialSrc" );
static const Slot sumRaxialSlot =
	initSymCompartmentCinfo()->getSlot( "raxial1.sumRaxialSrc" );
static const Slot sumRaxialSlotRequest =
	initSymCompartmentCinfo()->getSlot( "raxial1.sumRaxialRequestSrc");

static const Slot raxial2Slot =
	initSymCompartmentCinfo()->getSlot( "raxial2.Raxial2Src" );
static const Slot sumRaxial2Slot =
	initSymCompartmentCinfo()->getSlot( "raxial2.sumRaxial2Src" );
static const Slot sumRaxial2SlotRequest =
	initSymCompartmentCinfo()->getSlot( "raxial2.sumRaxial2RequestSrc" );

//////////////////////////////////////////////////////////////////
// Here we put the SymCompartment class functions.
//////////////////////////////////////////////////////////////////

SymCompartment::SymCompartment()
{
	;
}

//////////////////////////////////////////////////////////////////
// Compartment::Dest function definitions.
//////////////////////////////////////////////////////////////////

/*
void SymCompartment::innerProcessFunc( Element* e, ProcInfo p )
{
	A_ += Inject_ + sumInject_ + Em_ * invRm_; 
	if ( B_ > EPSILON ) {
		double x = exp( -B_ * p->dt_ / Cm_ );
		Vm_ = Vm_ * x + ( A_ / B_ )  * ( 1.0 - x );
	} else {
		Vm_ += ( A_ - Vm_ * B_ ) * p->dt_ / Cm_;
	}
	A_ = 0.0;
	B_ = invRm_; 
	Im_ = 0.0;
	sumInject_ = 0.0;
	// Send out the channel messages
	send1< double >( e, channelSlot, Vm_ );
	// Send out the message to any SpikeGens.
	send1< double >( e, VmSlot, Vm_ );
	// Send out the axial messages
	// send1< double >( e, axialSlot, Vm_ );
	// Send out the raxial messages
	// send2< double >( e, raxialSlot, Ra_, Vm_ );
}
*/

void SymCompartment::innerReinitFunc( Eref e, ProcInfo p )
{
	moose::Compartment::innerReinitFunc( e, p );
	coeff_ = 0.0;
	coeff2_ = 0.0;

	send0( e, sumRaxialSlotRequest );
	send0( e, sumRaxial2SlotRequest );

	coeff_ *= Ra_;
	coeff_ = ( 1 + coeff_ ) / 2.0;

	coeff2_ *= Ra_;
	coeff2_ = ( 1 + coeff2_ ) / 2.0;
}

void SymCompartment::sumRaxialRequest( const Conn* c )
{
	double Ra = static_cast< SymCompartment* >( c->data() )->Ra_;
	send1< double >( c->target(), sumRaxialSlot, Ra );
}

void SymCompartment::sumRaxial2Request( const Conn* c )
{
	double Ra = static_cast< SymCompartment* >( c->data() )->Ra_;
	send1< double >( c->target(), sumRaxial2Slot, Ra );
}

void SymCompartment::sumRaxial( const Conn* c, double Ra )
{
	static_cast< SymCompartment* >( c->data() )->coeff_ += 1.0 / Ra;
}

void SymCompartment::sumRaxial2( const Conn* c, double Ra )
{
	static_cast< SymCompartment* >( c->data() )->coeff2_ += 1.0 / Ra;
}

// Alternates with the 'process' message
void SymCompartment::innerInitFunc( Eref e, ProcInfo p )
{
	// Send out the raxial messages
	send2< double >( e, raxialSlot, Ra_, Vm_ );
	// Send out the raxial2 messages
	send2< double >( e, raxial2Slot, Ra_, Vm_ );
}

// This is called by the RaxialFunc, which is already defined in Compartment
void SymCompartment::innerRaxialFunc( double Ra, double Vm)
{
	Ra *= coeff_;
	A_ += Vm / Ra;
	B_ += 1.0 / Ra;
	Im_ += ( Vm - Vm_ ) / Ra;
}

void SymCompartment::innerRaxial2Func( double Ra, double Vm)
{
	Ra *= coeff2_;
	A_ += Vm / Ra;
	B_ += 1.0 / Ra;
	Im_ += ( Vm - Vm_ ) / Ra;
}

void SymCompartment::raxial2Func( const Conn* c, double Ra, double Vm)
{
	static_cast< SymCompartment* >( c->data() )->
			innerRaxial2Func( Ra, Vm );
}

/////////////////////////////////////////////////////////////////////

#ifdef DO_UNIT_TESTS
// Comment out this define if it takes too long (about 5 seconds on
// a modest machine, but could be much longer with valgrind)
// #define DO_SPATIAL_TESTS
#include "../element/Neutral.h"

void testSymCompartment()
{
	cout << "\nTesting SymCompartment" << flush;

	Element* n = Neutral::create( "Neutral", "n", Element::root()->id(), 
		Id::scratchId() );
	Element* c0 = Neutral::create( "SymCompartment", "c0", n->id(), 
		Id::scratchId() );
	ASSERT( c0 != 0, "creating symCompartment" );
	ProcInfoBase p;
	SetConn c( c0, 0 );
	p.dt_ = 0.002;
	moose::Compartment::setInject( &c, 1.0 );
	moose::Compartment::setRm( &c, 1.0 );
	moose::Compartment::setRa( &c, 0.0025 );
	moose::Compartment::setCm( &c, 1.0 );
	moose::Compartment::setEm( &c, 0.0 );
	moose::Compartment::setVm( &c, 0.0 );

	// Get rid of all the compartments.
	set( n, "destroy" );
}
#endif
