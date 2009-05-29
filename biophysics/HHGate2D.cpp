/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "../builtins/Interpol.h"
#include "../builtins/Interpol2D.h"
#include "HHGate.h"
#include "HHGate2D.h"

const Cinfo* initHHGate2DCinfo()
{
	static Finfo* gateShared[] =
	{
		new DestFinfo( "lookup", Ftype2< double, double >::global(),
						RFCAST( &HHGate2D::gateFunc ) ),
		new SrcFinfo( "gate", Ftype2< double, double >::global() ),
	};
	
	static Finfo* HHGate2DFinfos[] =
	{
	///////////////////////////////////////////////////////
	// Field definitions.
	///////////////////////////////////////////////////////
		new LookupFinfo( "A2D",
			LookupFtype< double, vector< double > >::global(),
			GFCAST( &HHGate2D::getAValue ),
			&dummyFunc ),
		new LookupFinfo( "B2D",
			LookupFtype< double, vector< double > >::global(),
			GFCAST( &HHGate2D::getBValue ),
			&dummyFunc ),
		
	///////////////////////////////////////////////////////
	// Shared definitions
	///////////////////////////////////////////////////////
		new SharedFinfo( "gate2D", gateShared, 
			sizeof( gateShared ) / sizeof( Finfo* ),
			"This is a shared message to communicate with the channel.\n"
			"Receives Vm and/or concentration \n"
			"Sends A and B from the respective table lookups." ),

	///////////////////////////////////////////////////////
	// MsgDest definitions
	///////////////////////////////////////////////////////
		new DestFinfo( "createInterpols", Ftype1< IdGenerator >::global(),
			RFCAST( &HHGate2D::createInterpols ),
			"Request the gate explicitly to create Interpols, with the given "
			"ids. This is used when the gate is a global object, and so the "
			"interpols need to be globals too. Comes in use in TABCREATE in the "
			"parallel context." ),
		new DestFinfo( "setupAlpha",
			Ftype1< vector< double > >::global(),
			RFCAST( &HHGate2D::setupAlpha ) ),
		new DestFinfo( "setupTau",
			Ftype1< vector< double > >::global(),
			RFCAST( &HHGate2D::setupTau ) ),
		new DestFinfo( "tweakAlpha", Ftype0::global(),
			&HHGate2D::tweakAlpha ),
		new DestFinfo( "tweakTau", Ftype0::global(),
			&HHGate2D::tweakTau ),
		new DestFinfo( "setupGate",
			Ftype1< vector< double > >::global(),
			RFCAST( &HHGate2D::setupGate ) ),
	};

	static string doc[] =
	{
		"Name", "HHGate2D",
		"Author", "Niraj Dudani, 2009, NCBS",
		"Description", "HHGate2D: Gate for Hodkgin-Huxley type channels, equivalent to the "
				"m and h terms on the Na squid channel and the n term on K. "
				"This takes the voltage and state variable from the channel, "
				"computes the new value of the state variable and a scaling, "
				"depending on gate power, for the conductance. These two "
				"terms are sent right back in a message to the channel.",
	};
	
	static Cinfo HHGate2DCinfo(
		doc,
		sizeof( doc ) / sizeof( string ),		
		initHHGateCinfo(),
		HHGate2DFinfos,
		sizeof( HHGate2DFinfos ) / sizeof( Finfo * ),
		ValueFtype1< HHGate2D >::global()
	);

	return &HHGate2DCinfo;
}

static const Cinfo* HHGate2DCinfo = initHHGate2DCinfo();

static const Slot gateSlot = initHHGate2DCinfo()->getSlot( "gate2D" );

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
double HHGate2D::getAValue( Eref e, const vector< double >& v )
{
	if ( v.size() < 2 ) {
		cerr << "Error: HHGate2D::getAValue: 2 real numbers needed to lookup 2D table.\n";
		return 0.0;
	}
	
	if ( v.size() > 2 ) {
		cerr << "Error: HHGate2D::getAValue: Only 2 real numbers needed to lookup 2D table. "
			"Using only first 2.\n";
	}
	
	HHGate2D* h = static_cast< HHGate2D* >( e.data() );
	return h->A_.innerLookup( v[ 0 ], v[ 1 ] );
}

double HHGate2D::getBValue( Eref e, const vector< double >& v )
{
	if ( v.size() < 2 ) {
		cerr << "Error: HHGate2D::getAValue: 2 real numbers needed to lookup 2D table.\n";
		return 0.0;
	}
	
	if ( v.size() > 2 ) {
		cerr << "Error: HHGate2D::getAValue: Only 2 real numbers needed to lookup 2D table. "
			"Using only first 2.\n";
	}
	
	HHGate2D* h = static_cast< HHGate2D* >( e.data() );
	return h->B_.innerLookup( v[ 0 ], v[ 1 ] );
}

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////
void HHGate2D::gateFunc( const Conn* c, double v1, double v2 )
{
	HHGate2D *h = static_cast< HHGate2D *>( c->data() );
	sendBack2< double, double >( c, gateSlot,
		h->A_.innerLookup( v1, v2 ) , h->B_.innerLookup( v1, v2 ) );
}

/**
 * Request the gate explicitly to create Interpols, with the given ids. This is
 * used when the gate is a global object, and so the interpols need to be
 * globals too. Comes in use in TABCREATE in the parallel context.
 */
void HHGate2D::createInterpols( const Conn* c, IdGenerator idGen )
{
	HHGate2D* h = static_cast< HHGate2D *>( c->data() );
	Eref e = c->target();
	
	const Cinfo* ic = initInterpol2DCinfo();
	
	// Here we must set the noDelFlag to 1 because these data
	// parts belong to the parent HHGate2D structure.
	Element* A = ic->create( 
		idGen.next(), "A", static_cast< void* >( &h->A_ ), 1 );
	e.add( "childSrc", A, "child" );

	Element* B = ic->create( 
		idGen.next(), "B", static_cast< void* >( &h->B_), 1 );
	e.add( "childSrc", B, "child" );
}
